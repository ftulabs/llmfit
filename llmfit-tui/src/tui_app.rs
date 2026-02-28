use llmfit_core::fit::{FitLevel, ModelFit, SortColumn, backend_compatible};
use llmfit_core::hardware::SystemSpecs;
use llmfit_core::models::ModelDatabase;
use llmfit_core::providers::{
    self, LlamaCppProvider, MlxProvider, ModelProvider, OllamaProvider, PullEvent, PullHandle,
};

use std::collections::{HashMap, HashSet};
use std::sync::mpsc;

use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Search,
    ProviderPopup,
    DownloadProviderPopup,
    ClusterPopup,
    ClusterAdd,
}

/// Tracks which field is active in the cluster add-node form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterField {
    Count,
    Name,
    Ram,
    Vram,
    Cores,
}

impl ClusterField {
    pub fn next(self) -> Self {
        match self {
            ClusterField::Count => ClusterField::Name,
            ClusterField::Name => ClusterField::Ram,
            ClusterField::Ram => ClusterField::Vram,
            ClusterField::Vram => ClusterField::Cores,
            ClusterField::Cores => ClusterField::Count,
        }
    }
    pub fn prev(self) -> Self {
        match self {
            ClusterField::Count => ClusterField::Cores,
            ClusterField::Name => ClusterField::Count,
            ClusterField::Ram => ClusterField::Name,
            ClusterField::Vram => ClusterField::Ram,
            ClusterField::Cores => ClusterField::Vram,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            ClusterField::Count => "Count",
            ClusterField::Name => "Name",
            ClusterField::Ram => "RAM",
            ClusterField::Vram => "VRAM",
            ClusterField::Cores => "Cores",
        }
    }
    pub fn all() -> &'static [ClusterField] {
        &[
            ClusterField::Count,
            ClusterField::Name,
            ClusterField::Ram,
            ClusterField::Vram,
            ClusterField::Cores,
        ]
    }
}

/// State for the add-node form in the cluster popup.
#[derive(Debug, Clone)]
pub struct ClusterForm {
    pub active_field: ClusterField,
    pub count: String,
    pub name: String,
    pub ram: String,
    pub vram: String,
    pub cores: String,
    pub error: Option<String>,
}

impl ClusterForm {
    pub fn new() -> Self {
        ClusterForm {
            active_field: ClusterField::Count,
            count: "1".to_string(),
            name: String::new(),
            ram: String::new(),
            vram: String::new(),
            cores: String::new(),
            error: None,
        }
    }

    pub fn field_value(&self, field: ClusterField) -> &str {
        match field {
            ClusterField::Count => &self.count,
            ClusterField::Name => &self.name,
            ClusterField::Ram => &self.ram,
            ClusterField::Vram => &self.vram,
            ClusterField::Cores => &self.cores,
        }
    }

    pub fn field_value_mut(&mut self, field: ClusterField) -> &mut String {
        match field {
            ClusterField::Count => &mut self.count,
            ClusterField::Name => &mut self.name,
            ClusterField::Ram => &mut self.ram,
            ClusterField::Vram => &mut self.vram,
            ClusterField::Cores => &mut self.cores,
        }
    }

    /// Build a node spec string and parse it.
    pub fn try_build(&self) -> Result<llmfit_core::cluster::ClusterNode, String> {
        if self.name.is_empty() {
            return Err("Name is required".to_string());
        }
        if self.ram.is_empty() {
            return Err("RAM is required".to_string());
        }

        let mut spec = format!("{}:{}:{}", self.count, self.name, self.ram);
        if !self.vram.is_empty() {
            spec.push(':');
            spec.push_str(&self.vram);
        }
        if !self.cores.is_empty() {
            spec.push(':');
            spec.push_str(&self.cores);
        }

        llmfit_core::cluster::ClusterConfig::parse_node(&spec)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitFilter {
    All,
    Perfect,
    Good,
    Marginal,
    TooTight,
    Runnable, // Perfect + Good + Marginal (excludes TooTight)
}

impl FitFilter {
    pub fn label(&self) -> &str {
        match self {
            FitFilter::All => "All",
            FitFilter::Perfect => "Perfect",
            FitFilter::Good => "Good",
            FitFilter::Marginal => "Marginal",
            FitFilter::TooTight => "Too Tight",
            FitFilter::Runnable => "Runnable",
        }
    }

    pub fn next(&self) -> Self {
        match self {
            FitFilter::All => FitFilter::Runnable,
            FitFilter::Runnable => FitFilter::Perfect,
            FitFilter::Perfect => FitFilter::Good,
            FitFilter::Good => FitFilter::Marginal,
            FitFilter::Marginal => FitFilter::TooTight,
            FitFilter::TooTight => FitFilter::All,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadProvider {
    Ollama,
    LlamaCpp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadCapability {
    Unknown,
    None,
    Ollama,
    LlamaCpp,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivePullProvider {
    Ollama,
    Mlx,
    LlamaCpp,
}

impl ActivePullProvider {
    fn label(self) -> &'static str {
        match self {
            ActivePullProvider::Ollama => "Ollama",
            ActivePullProvider::Mlx => "MLX",
            ActivePullProvider::LlamaCpp => "llama.cpp",
        }
    }
}

pub struct App {
    pub should_quit: bool,
    pub input_mode: InputMode,
    pub search_query: String,
    pub cursor_position: usize,

    // Data
    pub specs: SystemSpecs,
    pub all_fits: Vec<ModelFit>,
    pub filtered_fits: Vec<usize>, // indices into all_fits
    pub providers: Vec<String>,
    pub selected_providers: Vec<bool>,

    // Filters
    pub fit_filter: FitFilter,
    pub installed_first: bool,
    pub sort_column: SortColumn,

    // Table state
    pub selected_row: usize,

    // Detail view
    pub show_detail: bool,

    // Provider popup
    pub provider_cursor: usize,
    pub download_provider_cursor: usize,
    pub download_provider_options: Vec<DownloadProvider>,
    pub download_provider_model: Option<String>,

    // Provider state
    pub ollama_available: bool,
    pub ollama_binary_available: bool,
    pub ollama_installed: HashSet<String>,
    ollama: OllamaProvider,
    pub mlx_available: bool,
    pub mlx_installed: HashSet<String>,
    mlx: MlxProvider,
    pub llamacpp_available: bool,
    pub llamacpp_installed: HashSet<String>,
    llamacpp: LlamaCppProvider,

    // Download state
    pub pull_active: Option<PullHandle>,
    pub pull_status: Option<String>,
    pub pull_percent: Option<f64>,
    pub pull_model_name: Option<String>,
    pull_provider: Option<ActivePullProvider>,
    pub download_capabilities: HashMap<String, DownloadCapability>,
    download_capability_inflight: HashSet<String>,
    download_capability_tx: mpsc::Sender<(String, DownloadCapability)>,
    download_capability_rx: mpsc::Receiver<(String, DownloadCapability)>,
    /// Animation frame counter, incremented every tick while pulling.
    pub tick_count: u64,
    /// When true, the next 'd' press will confirm and start the download.
    pub confirm_download: bool,

    // Theme
    pub theme: Theme,

    /// How many models we silently dropped because they can't run on this
    /// hardware — shown in the system bar so users aren't left wondering
    /// why the list looks shorter than expected.
    pub backend_hidden_count: usize,

    // Cluster configuration
    pub cluster_nodes: Vec<llmfit_core::cluster::ClusterNode>,
    pub cluster_active: bool,
    pub cluster_cursor: usize,
    pub cluster_form: Option<ClusterForm>,
    pub context_limit: Option<u32>,
    local_specs: SystemSpecs,
}

impl App {
    pub fn with_specs(specs: SystemSpecs) -> Self {
        Self::with_specs_and_context(specs, None)
    }

    pub fn with_specs_and_context(specs: SystemSpecs, context_limit: Option<u32>) -> Self {
        let db = ModelDatabase::new();

        // Detect Ollama
        let ollama = OllamaProvider::new();
        let (ollama_available, ollama_installed) = ollama.detect_with_installed();
        let ollama_binary_available = command_exists("ollama");

        // Detect MLX
        let mlx = MlxProvider::new();
        let (mlx_available, mlx_installed) = mlx.detect_with_installed();

        // Detect llama.cpp
        let llamacpp = LlamaCppProvider::new();
        let llamacpp_available = llamacpp.is_available();
        let llamacpp_installed = llamacpp.installed_models();

        // Track how many we're skipping so the UI can surface it.
        let backend_hidden_count = db
            .get_all_models()
            .iter()
            .filter(|m| !backend_compatible(m, &specs))
            .count();

        // Only analyze models that can actually run on this hardware.
        let mut all_fits: Vec<ModelFit> = db
            .get_all_models()
            .iter()
            .filter(|m| backend_compatible(m, &specs))
            .map(|m| {
                let mut fit = ModelFit::analyze_with_context_limit(m, &specs, context_limit);
                fit.installed = providers::is_model_installed(&m.name, &ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &mlx_installed)
                    || providers::is_model_installed_llamacpp(&m.name, &llamacpp_installed);
                fit
            })
            .collect();

        // Sort by fit level then RAM usage
        all_fits = llmfit_core::fit::rank_models_by_fit(all_fits);

        // Extract unique providers
        let mut model_providers: Vec<String> = all_fits
            .iter()
            .map(|f| f.model.provider.clone())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        model_providers.sort();

        let selected_providers = vec![true; model_providers.len()];

        let filtered_count = all_fits.len();

        let (download_capability_tx, download_capability_rx) = mpsc::channel();

        let local_specs = specs.clone();
        let mut app = App {
            should_quit: false,
            input_mode: InputMode::Normal,
            search_query: String::new(),
            cursor_position: 0,
            specs,
            all_fits,
            filtered_fits: (0..filtered_count).collect(),
            providers: model_providers,
            selected_providers,
            fit_filter: FitFilter::All,
            installed_first: false,
            sort_column: SortColumn::Score,
            selected_row: 0,
            show_detail: false,
            provider_cursor: 0,
            download_provider_cursor: 0,
            download_provider_options: Vec::new(),
            download_provider_model: None,
            ollama_available,
            ollama_binary_available,
            ollama_installed,
            ollama,
            mlx_available,
            mlx_installed,
            mlx,
            llamacpp_available,
            llamacpp_installed,
            llamacpp,
            pull_active: None,
            pull_status: None,
            pull_percent: None,
            pull_model_name: None,
            pull_provider: None,
            download_capabilities: HashMap::new(),
            download_capability_inflight: HashSet::new(),
            download_capability_tx,
            download_capability_rx,
            tick_count: 0,
            confirm_download: false,
            theme: Theme::load(),
            backend_hidden_count,
            cluster_nodes: Vec::new(),
            cluster_active: false,
            cluster_cursor: 0,
            cluster_form: None,
            context_limit,
            local_specs,
        };

        app.apply_filters();
        app.enqueue_capability_probes_for_visible(24);
        app
    }

    // ── Cluster configuration ────────────────────────────────────

    pub fn open_cluster_popup(&mut self) {
        self.input_mode = InputMode::ClusterPopup;
    }

    pub fn close_cluster_popup(&mut self) {
        self.input_mode = InputMode::Normal;
        self.cluster_form = None;
    }

    pub fn cluster_popup_up(&mut self) {
        if self.cluster_cursor > 0 {
            self.cluster_cursor -= 1;
        }
    }

    pub fn cluster_popup_down(&mut self) {
        if !self.cluster_nodes.is_empty() && self.cluster_cursor < self.cluster_nodes.len() - 1 {
            self.cluster_cursor += 1;
        }
    }

    pub fn start_cluster_add(&mut self) {
        self.cluster_form = Some(ClusterForm::new());
        self.input_mode = InputMode::ClusterAdd;
    }

    pub fn cancel_cluster_add(&mut self) {
        self.cluster_form = None;
        self.input_mode = InputMode::ClusterPopup;
    }

    pub fn cluster_form_next_field(&mut self) {
        if let Some(form) = &mut self.cluster_form {
            form.active_field = form.active_field.next();
            form.error = None;
        }
    }

    pub fn cluster_form_prev_field(&mut self) {
        if let Some(form) = &mut self.cluster_form {
            form.active_field = form.active_field.prev();
            form.error = None;
        }
    }

    pub fn cluster_form_input(&mut self, c: char) {
        if let Some(form) = &mut self.cluster_form {
            let field = form.active_field;
            form.field_value_mut(field).push(c);
            form.error = None;
        }
    }

    pub fn cluster_form_backspace(&mut self) {
        if let Some(form) = &mut self.cluster_form {
            let field = form.active_field;
            form.field_value_mut(field).pop();
            form.error = None;
        }
    }

    pub fn cluster_form_submit(&mut self) {
        let Some(form) = &self.cluster_form else {
            return;
        };
        match form.try_build() {
            Ok(node) => {
                self.cluster_nodes.push(node);
                self.cluster_form = None;
                self.input_mode = InputMode::ClusterPopup;
                if self.cluster_active {
                    self.recalculate_cluster();
                }
            }
            Err(e) => {
                if let Some(form) = &mut self.cluster_form {
                    form.error = Some(e);
                }
            }
        }
    }

    pub fn cluster_delete_node(&mut self) {
        if !self.cluster_nodes.is_empty() {
            self.cluster_nodes.remove(self.cluster_cursor);
            if self.cluster_cursor > 0 && self.cluster_cursor >= self.cluster_nodes.len() {
                self.cluster_cursor = self.cluster_nodes.len().saturating_sub(1);
            }
            if self.cluster_active {
                if self.cluster_nodes.is_empty() {
                    self.deactivate_cluster();
                } else {
                    self.recalculate_cluster();
                }
            }
        }
    }

    pub fn toggle_cluster(&mut self) {
        if self.cluster_active {
            self.deactivate_cluster();
        } else if !self.cluster_nodes.is_empty() {
            self.activate_cluster();
        }
    }

    fn activate_cluster(&mut self) {
        self.cluster_active = true;
        self.recalculate_cluster();
    }

    fn deactivate_cluster(&mut self) {
        self.cluster_active = false;
        self.specs = self.local_specs.clone();
        self.recalculate_fits();
    }

    fn recalculate_cluster(&mut self) {
        let mut config = llmfit_core::cluster::ClusterConfig::new();
        for node in &self.cluster_nodes {
            config.add_node(node.clone());
        }
        let (specs, _) = llmfit_core::cluster::analyze_cluster(
            &config,
            &ModelDatabase::new(),
            self.context_limit,
        );
        self.specs = specs;
        self.recalculate_fits();
    }

    fn recalculate_fits(&mut self) {
        let db = ModelDatabase::new();
        let mut all_fits: Vec<ModelFit> = db
            .get_all_models()
            .iter()
            .map(|m| {
                let mut fit =
                    ModelFit::analyze_with_context_limit(m, &self.specs, self.context_limit);
                fit.installed = providers::is_model_installed(&m.name, &self.ollama_installed)
                    || providers::is_model_installed_mlx(&m.name, &self.mlx_installed);
                fit
            })
            .collect();

        // Apply cluster distribution adjustments when in cluster mode
        if self.cluster_active {
            let mut config = llmfit_core::cluster::ClusterConfig::new();
            for node in &self.cluster_nodes {
                config.add_node(node.clone());
            }
            for fit in &mut all_fits {
                llmfit_core::cluster::adjust_for_cluster(fit, &config);
            }
        }

        all_fits = llmfit_core::fit::rank_models_by_fit_opts_col(
            all_fits,
            self.installed_first,
            self.sort_column,
        );
        self.all_fits = all_fits;
        self.apply_filters();
    }

    pub fn apply_filters(&mut self) {
        let query = self.search_query.to_lowercase();
        // Split query into space-separated terms for fuzzy matching
        let terms: Vec<&str> = query.split_whitespace().collect();

        self.filtered_fits = self
            .all_fits
            .iter()
            .enumerate()
            .filter(|(_, fit)| {
                // Search filter: all terms must match (fuzzy/AND logic)
                let matches_search = if terms.is_empty() {
                    true
                } else {
                    // Combine all searchable fields into one string
                    let searchable = format!(
                        "{} {} {} {}",
                        fit.model.name.to_lowercase(),
                        fit.model.provider.to_lowercase(),
                        fit.model.parameter_count.to_lowercase(),
                        fit.model.use_case.to_lowercase()
                    );
                    // All terms must be present (AND logic)
                    terms.iter().all(|term| searchable.contains(term))
                };

                // Provider filter
                let provider_idx = self.providers.iter().position(|p| p == &fit.model.provider);
                let matches_provider = provider_idx
                    .map(|idx| self.selected_providers[idx])
                    .unwrap_or(true);

                // Fit filter
                let matches_fit = match self.fit_filter {
                    FitFilter::All => true,
                    FitFilter::Perfect => fit.fit_level == FitLevel::Perfect,
                    FitFilter::Good => fit.fit_level == FitLevel::Good,
                    FitFilter::Marginal => fit.fit_level == FitLevel::Marginal,
                    FitFilter::TooTight => fit.fit_level == FitLevel::TooTight,
                    FitFilter::Runnable => fit.fit_level != FitLevel::TooTight,
                };

                matches_search && matches_provider && matches_fit
            })
            .map(|(i, _)| i)
            .collect();

        // Clamp selection
        if self.filtered_fits.is_empty() {
            self.selected_row = 0;
        } else if self.selected_row >= self.filtered_fits.len() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn selected_fit(&self) -> Option<&ModelFit> {
        self.filtered_fits
            .get(self.selected_row)
            .map(|&idx| &self.all_fits[idx])
    }

    pub fn move_up(&mut self) {
        self.confirm_download = false;
        if self.selected_row > 0 {
            self.selected_row -= 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn move_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() && self.selected_row < self.filtered_fits.len() - 1 {
            self.selected_row += 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_up(&mut self) {
        self.confirm_download = false;
        self.selected_row = self.selected_row.saturating_sub(10);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn page_down(&mut self) {
        self.confirm_download = false;
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 10).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_up(&mut self) {
        self.selected_row = self.selected_row.saturating_sub(5);
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn half_page_down(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = (self.selected_row + 5).min(self.filtered_fits.len() - 1);
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn home(&mut self) {
        self.selected_row = 0;
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn end(&mut self) {
        if !self.filtered_fits.is_empty() {
            self.selected_row = self.filtered_fits.len() - 1;
        }
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn cycle_fit_filter(&mut self) {
        self.fit_filter = self.fit_filter.next();
        self.apply_filters();
    }

    pub fn cycle_sort_column(&mut self) {
        self.sort_column = self.sort_column.next();
        self.re_sort();
    }

    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
        self.theme.save();
    }

    pub fn enter_search(&mut self) {
        self.input_mode = InputMode::Search;
    }

    pub fn exit_search(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn search_input(&mut self, c: char) {
        self.search_query.insert(self.cursor_position, c);
        self.cursor_position += 1;
        self.apply_filters();
    }

    pub fn search_backspace(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn search_delete(&mut self) {
        if self.cursor_position < self.search_query.len() {
            self.search_query.remove(self.cursor_position);
            self.apply_filters();
        }
    }

    pub fn clear_search(&mut self) {
        self.search_query.clear();
        self.cursor_position = 0;
        self.apply_filters();
    }

    pub fn toggle_detail(&mut self) {
        self.show_detail = !self.show_detail;
    }

    pub fn open_provider_popup(&mut self) {
        self.input_mode = InputMode::ProviderPopup;
        // Don't reset cursor -- keep it where it was last time
    }

    pub fn close_provider_popup(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn provider_popup_up(&mut self) {
        if self.provider_cursor > 0 {
            self.provider_cursor -= 1;
        }
    }

    pub fn provider_popup_down(&mut self) {
        if self.provider_cursor + 1 < self.providers.len() {
            self.provider_cursor += 1;
        }
    }

    pub fn provider_popup_toggle(&mut self) {
        if self.provider_cursor < self.selected_providers.len() {
            self.selected_providers[self.provider_cursor] =
                !self.selected_providers[self.provider_cursor];
            self.apply_filters();
        }
    }

    pub fn provider_popup_select_all(&mut self) {
        let all_selected = self.selected_providers.iter().all(|&s| s);
        let new_val = !all_selected;
        for s in &mut self.selected_providers {
            *s = new_val;
        }
        self.apply_filters();
    }

    pub fn toggle_installed_first(&mut self) {
        self.installed_first = !self.installed_first;
        self.re_sort();
    }

    /// Re-sort all_fits using current sort column and installed_first preference, then refilter.
    fn re_sort(&mut self) {
        let fits = std::mem::take(&mut self.all_fits);
        self.all_fits = llmfit_core::fit::rank_models_by_fit_opts_col(
            fits,
            self.installed_first,
            self.sort_column,
        );
        self.apply_filters();
    }

    /// Start pulling the currently selected model via the best available provider.
    pub fn start_download(&mut self) {
        let any_available = self.ollama_available || self.mlx_available || self.llamacpp_available;
        if !any_available {
            self.pull_status = Some("No provider available (Ollama/MLX/llama.cpp)".to_string());
            return;
        }
        if self.pull_active.is_some() {
            return; // already pulling
        }
        let Some(fit) = self.selected_fit() else {
            return;
        };
        if fit.installed {
            self.pull_status = Some("Already installed".to_string());
            return;
        }
        let model_name = fit.model.name.clone();

        // Choose provider based on runtime
        let use_mlx = fit.runtime == llmfit_core::fit::InferenceRuntime::Mlx && self.mlx_available;

        if use_mlx {
            self.start_mlx_download(model_name);
            return;
        }

        let download_options = self.available_download_providers(&model_name);
        if !download_options.is_empty() {
            self.open_download_provider_popup(model_name, download_options);
        } else {
            self.pull_status = Some("No compatible provider available for this model".to_string());
        }
    }

    fn start_mlx_download(&mut self, model_name: String) {
        let tag = providers::mlx_pull_tag(&model_name);
        match self.mlx.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling mlx-community/{}...", tag));
                self.pull_percent = None;
                self.pull_provider = Some(ActivePullProvider::Mlx);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("MLX pull failed: {}", e));
            }
        }
    }

    fn start_download_with_provider(&mut self, model_name: String, provider: DownloadProvider) {
        match provider {
            DownloadProvider::Ollama => self.start_ollama_download(model_name),
            DownloadProvider::LlamaCpp => self.start_llamacpp_download_for_model(model_name),
        }
    }

    fn start_ollama_download(&mut self, model_name: String) {
        let Some(tag) = providers::ollama_pull_tag(&model_name) else {
            self.pull_status = Some("Not available in Ollama registry".to_string());
            return;
        };
        match self.ollama.start_pull(&tag) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Pulling {}...", tag));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::Ollama);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("Pull failed: {}", e));
            }
        }
    }

    /// Start downloading a GGUF model via the llama.cpp provider.
    fn start_llamacpp_download_for_model(&mut self, model_name: String) {
        let Some(repo) = providers::first_existing_gguf_repo(&model_name) else {
            self.pull_status = Some("No GGUF repo found in remote registry".to_string());
            return;
        };

        match self.llamacpp.start_pull(&repo) {
            Ok(handle) => {
                self.pull_model_name = Some(model_name);
                self.pull_status = Some(format!("Downloading GGUF from {}...", repo));
                self.pull_percent = Some(0.0);
                self.pull_provider = Some(ActivePullProvider::LlamaCpp);
                self.pull_active = Some(handle);
            }
            Err(e) => {
                self.pull_status = Some(format!("GGUF download failed: {}", e));
            }
        }
    }

    /// Poll the active pull for progress. Called each TUI tick.
    pub fn tick_pull(&mut self) {
        self.enqueue_capability_probes_for_visible(24);
        self.tick_download_capability();
        if self.pull_active.is_some() {
            self.tick_count = self.tick_count.wrapping_add(1);
        }
        let Some(handle) = &self.pull_active else {
            return;
        };
        // Drain all available events
        loop {
            match handle.receiver.try_recv() {
                Ok(PullEvent::Progress { status, percent }) => {
                    if let Some(p) = percent {
                        self.pull_percent = Some(p);
                    }
                    self.pull_status = Some(status);
                }
                Ok(PullEvent::Done) => {
                    let done_msg = if let Some(provider) = self.pull_provider {
                        format!("Download complete via {}!", provider.label())
                    } else {
                        "Download complete!".to_string()
                    };
                    self.pull_status = Some(done_msg);
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    // Refresh installed models
                    self.refresh_installed();
                    return;
                }
                Ok(PullEvent::Error(e)) => {
                    self.pull_status = Some(format!("Error: {}", e));
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    return;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.pull_status = Some("Pull ended".to_string());
                    self.pull_percent = None;
                    self.pull_active = None;
                    self.pull_provider = None;
                    self.refresh_installed();
                    return;
                }
            }
        }
    }

    fn available_download_providers(&self, model_name: &str) -> Vec<DownloadProvider> {
        let mut providers_for_model = Vec::new();
        if providers::has_ollama_mapping(model_name)
            && (self.ollama_available || self.ollama_binary_available)
        {
            providers_for_model.push(DownloadProvider::Ollama);
        }
        if self.llamacpp_available && providers::first_existing_gguf_repo(model_name).is_some() {
            providers_for_model.push(DownloadProvider::LlamaCpp);
        }
        providers_for_model
    }

    fn open_download_provider_popup(&mut self, model_name: String, options: Vec<DownloadProvider>) {
        self.download_provider_model = Some(model_name);
        self.download_provider_options = options;
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::DownloadProviderPopup;
        self.pull_status = Some("Choose download provider and press Enter".to_string());
    }

    pub fn close_download_provider_popup(&mut self) {
        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.pull_status = Some("Download cancelled".to_string());
    }

    pub fn download_provider_popup_up(&mut self) {
        if self.download_provider_cursor > 0 {
            self.download_provider_cursor -= 1;
        }
    }

    pub fn download_provider_popup_down(&mut self) {
        if self.download_provider_cursor + 1 < self.download_provider_options.len() {
            self.download_provider_cursor += 1;
        }
    }

    pub fn confirm_download_provider_selection(&mut self) {
        let Some(model_name) = self.download_provider_model.clone() else {
            self.input_mode = InputMode::Normal;
            return;
        };
        let Some(provider) = self
            .download_provider_options
            .get(self.download_provider_cursor)
            .copied()
        else {
            self.close_download_provider_popup();
            return;
        };

        self.download_provider_model = None;
        self.download_provider_options.clear();
        self.download_provider_cursor = 0;
        self.input_mode = InputMode::Normal;
        self.start_download_with_provider(model_name, provider);
    }

    /// Re-query all providers for installed models and update all_fits.
    pub fn refresh_installed(&mut self) {
        self.ollama_installed = self.ollama.installed_models();
        self.mlx_installed = self.mlx.installed_models();
        self.llamacpp_installed = self.llamacpp.installed_models();
        for fit in &mut self.all_fits {
            fit.installed = providers::is_model_installed(&fit.model.name, &self.ollama_installed)
                || providers::is_model_installed_mlx(&fit.model.name, &self.mlx_installed)
                || providers::is_model_installed_llamacpp(
                    &fit.model.name,
                    &self.llamacpp_installed,
                );
        }
        self.re_sort();
        self.enqueue_capability_probes_for_visible(24);
    }

    pub fn download_capability_for(&self, model_name: &str) -> DownloadCapability {
        self.download_capabilities
            .get(model_name)
            .copied()
            .unwrap_or(DownloadCapability::Unknown)
    }

    pub fn enqueue_capability_probes_for_visible(&mut self, window: usize) {
        if self.filtered_fits.is_empty() {
            return;
        }
        let start = self.selected_row.saturating_sub(window / 2);
        let end = (start + window).min(self.filtered_fits.len());
        for idx in start..end {
            if let Some(&fit_idx) = self.filtered_fits.get(idx) {
                let model_name = self.all_fits[fit_idx].model.name.clone();
                self.enqueue_capability_probe(model_name);
            }
        }
    }

    fn enqueue_capability_probe(&mut self, model_name: String) {
        if self.download_capabilities.contains_key(&model_name)
            || self.download_capability_inflight.contains(&model_name)
            || self.download_capability_inflight.len() >= 12
        {
            return;
        }
        self.download_capability_inflight.insert(model_name.clone());

        let tx = self.download_capability_tx.clone();
        let ollama_runtime_available = self.ollama_available || self.ollama_binary_available;
        let llamacpp_available = self.llamacpp_available;
        std::thread::spawn(move || {
            let has_ollama = ollama_runtime_available && providers::has_ollama_mapping(&model_name);
            let mut has_llamacpp = false;
            if llamacpp_available {
                has_llamacpp = providers::first_existing_gguf_repo(&model_name).is_some();
            }

            let capability = match (has_ollama, has_llamacpp) {
                (true, true) => DownloadCapability::Both,
                (true, false) => DownloadCapability::Ollama,
                (false, true) => DownloadCapability::LlamaCpp,
                (false, false) => DownloadCapability::None,
            };
            let _ = tx.send((model_name, capability));
        });
    }

    fn tick_download_capability(&mut self) {
        loop {
            match self.download_capability_rx.try_recv() {
                Ok((name, capability)) => {
                    self.download_capability_inflight.remove(&name);
                    self.download_capabilities.insert(name, capability);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }
}

fn command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

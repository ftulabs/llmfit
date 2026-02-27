use crate::fit::{self, ModelFit, RunMode};
use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs};
use crate::models::ModelDatabase;

/// A single node type in a cluster configuration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterNode {
    pub name: String,
    pub count: u32,
    pub ram_gb: f64,
    pub vram_gb: Option<f64>,
    pub cpu_cores: u32,
}

/// A cluster configuration composed of one or more node types.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    pub nodes: Vec<ClusterNode>,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterConfig {
    pub fn new() -> Self {
        ClusterConfig { nodes: Vec::new() }
    }

    pub fn add_node(&mut self, node: ClusterNode) {
        self.nodes.push(node);
    }

    pub fn total_vram_gb(&self) -> f64 {
        self.nodes
            .iter()
            .filter_map(|n| n.vram_gb.map(|v| v * n.count as f64))
            .sum()
    }

    pub fn total_ram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.ram_gb * n.count as f64).sum()
    }

    pub fn total_gpu_count(&self) -> u32 {
        self.nodes
            .iter()
            .filter(|n| n.vram_gb.is_some())
            .map(|n| n.count)
            .sum()
    }

    pub fn total_cpu_cores(&self) -> u32 {
        self.nodes.iter().map(|n| n.cpu_cores * n.count).sum()
    }

    pub fn total_node_count(&self) -> u32 {
        self.nodes.iter().map(|n| n.count).sum()
    }

    /// Build a synthetic `SystemSpecs` representing the cluster's aggregate resources.
    pub fn to_system_specs(&self) -> SystemSpecs {
        let total_vram = self.total_vram_gb();
        let total_ram = self.total_ram_gb();
        let total_cores = self.total_cpu_cores() as usize;
        let gpu_count = self.total_gpu_count();
        let has_gpu = gpu_count > 0;

        let primary_gpu = self
            .nodes
            .iter()
            .filter(|n| n.vram_gb.is_some())
            .max_by(|a, b| {
                a.vram_gb
                    .partial_cmp(&b.vram_gb)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        let gpu_vram_gb = primary_gpu.and_then(|g| g.vram_gb);
        let gpu_name = primary_gpu.map(|g| g.name.clone());

        let gpus: Vec<GpuInfo> = self
            .nodes
            .iter()
            .filter(|n| n.vram_gb.is_some())
            .map(|n| GpuInfo {
                name: n.name.clone(),
                vram_gb: n.vram_gb,
                backend: GpuBackend::Cuda,
                count: n.count,
                unified_memory: false,
            })
            .collect();

        SystemSpecs {
            total_ram_gb: total_ram,
            available_ram_gb: total_ram * 0.9,
            total_cpu_cores: total_cores,
            cpu_name: format!(
                "Cluster ({} nodes, {} cores)",
                self.total_node_count(),
                total_cores
            ),
            has_gpu,
            gpu_vram_gb,
            total_gpu_vram_gb: if has_gpu { Some(total_vram) } else { None },
            gpu_name,
            gpu_count,
            unified_memory: false,
            backend: if has_gpu {
                GpuBackend::Cuda
            } else {
                GpuBackend::CpuX86
            },
            gpus,
        }
    }

    /// Parse a node specification string.
    ///
    /// Format: `count:name:ram[:vram[:cores]]`
    ///
    /// - `"2:A100-40GB:128G:40G"` — 2 GPU nodes, 128 GB RAM, 40 GB VRAM each
    /// - `"8:cpu:16G"` — 8 CPU-only nodes, 16 GB RAM each
    /// - `"4:H100:256G:80G:64"` — 4 GPU nodes, 256 GB RAM, 80 GB VRAM, 64 cores each
    pub fn parse_node(spec: &str) -> Result<ClusterNode, String> {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() < 3 {
            return Err(format!(
                "Invalid node spec '{}'. Expected format: count:name:ram[:vram[:cores]]",
                spec
            ));
        }

        let count: u32 = parts[0]
            .parse()
            .map_err(|_| format!("Invalid count '{}' in node spec", parts[0]))?;
        if count == 0 {
            return Err("Node count must be at least 1".to_string());
        }

        let name = parts[1].to_string();
        let ram_gb = parse_memory(parts[2])?;

        let is_cpu = name.eq_ignore_ascii_case("cpu");

        let vram_gb = if is_cpu {
            None
        } else if parts.len() > 3 {
            Some(parse_memory(parts[3])?)
        } else {
            return Err(format!(
                "GPU node '{}' requires VRAM. Format: count:name:ram:vram[:cores]",
                name
            ));
        };

        let cores_idx = if is_cpu { 3 } else { 4 };
        let cpu_cores = if parts.len() > cores_idx {
            parts[cores_idx]
                .parse()
                .map_err(|_| format!("Invalid cores '{}' in node spec", parts[cores_idx]))?
        } else if vram_gb.is_some() {
            32
        } else {
            8
        };

        Ok(ClusterNode {
            name,
            count,
            ram_gb,
            vram_gb,
            cpu_cores,
        })
    }
}

fn parse_memory(s: &str) -> Result<f64, String> {
    crate::hardware::parse_memory_size(s).ok_or_else(|| {
        format!(
            "Invalid memory size '{}'. Expected format: 32G, 32000M, 1.5T",
            s
        )
    })
}

/// Analyze all models in the database against a cluster configuration.
///
/// Returns the synthetic system specs and a ranked list of model fits.
/// Models that require distribution across multiple GPUs get a
/// `RunMode::Distributed` designation and a communication overhead penalty.
pub fn analyze_cluster(
    config: &ClusterConfig,
    db: &ModelDatabase,
    context_limit: Option<u32>,
) -> (SystemSpecs, Vec<ModelFit>) {
    let specs = config.to_system_specs();

    let mut fits: Vec<ModelFit> = db
        .get_all_models()
        .iter()
        .map(|m| {
            let mut fit = ModelFit::analyze_with_context_limit(m, &specs, context_limit);
            adjust_for_cluster(&mut fit, config);
            fit
        })
        .collect();

    fits = fit::rank_models_by_fit(fits);
    (specs, fits)
}

/// Post-process a ModelFit to account for cluster distribution.
pub fn adjust_for_cluster(fit: &mut ModelFit, config: &ClusterConfig) {
    let total_gpu_count = config.total_gpu_count();
    if total_gpu_count == 0 {
        return;
    }

    let is_gpu_path = matches!(
        fit.run_mode,
        RunMode::Gpu | RunMode::MoeOffload | RunMode::Distributed
    );
    if !is_gpu_path {
        return;
    }

    // Find the largest per-card VRAM across all GPU node types
    let max_per_card_vram: f64 = config
        .nodes
        .iter()
        .filter_map(|n| n.vram_gb)
        .fold(0.0_f64, f64::max);

    if max_per_card_vram <= 0.0 {
        return;
    }

    // If the model needs more VRAM than a single card, it's distributed
    if fit.memory_required_gb > max_per_card_vram {
        let gpus_needed = (fit.memory_required_gb / max_per_card_vram).ceil() as u32;
        let gpus_used = gpus_needed.min(total_gpu_count);

        fit.run_mode = RunMode::Distributed;

        fit.notes.push(format!(
            "Distributed: model split across {} GPUs via tensor parallelism",
            gpus_used
        ));

        // Multi-GPU communication overhead: ~12% per additional GPU
        let overhead = 1.0 - (0.12 * (gpus_used - 1) as f64).min(0.5);
        fit.estimated_tps *= overhead;
        fit.notes.push(format!(
            "Cluster efficiency: {:.0}% (multi-GPU communication overhead)",
            overhead * 100.0
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_node_gpu() {
        let node = ClusterConfig::parse_node("2:A100-40GB:128G:40G").unwrap();
        assert_eq!(node.count, 2);
        assert_eq!(node.name, "A100-40GB");
        assert!((node.ram_gb - 128.0).abs() < 0.01);
        assert!((node.vram_gb.unwrap() - 40.0).abs() < 0.01);
        assert_eq!(node.cpu_cores, 32);
    }

    #[test]
    fn test_parse_node_cpu() {
        let node = ClusterConfig::parse_node("8:cpu:16G").unwrap();
        assert_eq!(node.count, 8);
        assert_eq!(node.name, "cpu");
        assert!((node.ram_gb - 16.0).abs() < 0.01);
        assert!(node.vram_gb.is_none());
        assert_eq!(node.cpu_cores, 8);
    }

    #[test]
    fn test_parse_node_with_cores() {
        let node = ClusterConfig::parse_node("4:H100:256G:80G:64").unwrap();
        assert_eq!(node.count, 4);
        assert_eq!(node.name, "H100");
        assert!((node.ram_gb - 256.0).abs() < 0.01);
        assert!((node.vram_gb.unwrap() - 80.0).abs() < 0.01);
        assert_eq!(node.cpu_cores, 64);
    }

    #[test]
    fn test_parse_node_invalid() {
        assert!(ClusterConfig::parse_node("bad").is_err());
        assert!(ClusterConfig::parse_node("0:cpu:16G").is_err());
        assert!(ClusterConfig::parse_node("2:H100:128G").is_err()); // GPU needs VRAM
    }

    #[test]
    fn test_cluster_totals() {
        let mut config = ClusterConfig::new();
        config.add_node(ClusterConfig::parse_node("2:A100-40GB:128G:40G").unwrap());
        config.add_node(ClusterConfig::parse_node("8:cpu:16G").unwrap());

        assert!((config.total_vram_gb() - 80.0).abs() < 0.01);
        assert!((config.total_ram_gb() - 384.0).abs() < 0.01);
        assert_eq!(config.total_gpu_count(), 2);
        assert_eq!(config.total_node_count(), 10);
    }

    #[test]
    fn test_to_system_specs() {
        let mut config = ClusterConfig::new();
        config.add_node(ClusterConfig::parse_node("2:A100-40GB:128G:40G").unwrap());
        config.add_node(ClusterConfig::parse_node("8:cpu:16G").unwrap());

        let specs = config.to_system_specs();
        assert!(specs.has_gpu);
        assert_eq!(specs.gpu_count, 2);
        assert!((specs.total_gpu_vram_gb.unwrap() - 80.0).abs() < 0.01);
        assert!((specs.total_ram_gb - 384.0).abs() < 0.01);
        assert!((specs.available_ram_gb - 345.6).abs() < 0.1);
    }

    #[test]
    fn test_analyze_cluster_produces_results() {
        let mut config = ClusterConfig::new();
        config.add_node(ClusterConfig::parse_node("2:A100-40GB:128G:40G").unwrap());
        config.add_node(ClusterConfig::parse_node("8:cpu:16G").unwrap());

        let db = ModelDatabase::new();
        let (specs, fits) = analyze_cluster(&config, &db, None);
        assert!(specs.has_gpu);
        assert!(!fits.is_empty());
    }
}

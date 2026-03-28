use crate::hardware::{GpuBackend, GpuInfo, SystemSpecs};

/// Network interconnect between cluster nodes — affects distributed inference speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, Default)]
pub enum Interconnect {
    InfiniBand,
    Ethernet10G,
    #[default]
    Ethernet1G,
}

impl Interconnect {
    pub fn label(&self) -> &'static str {
        match self {
            Interconnect::InfiniBand => "InfiniBand",
            Interconnect::Ethernet10G => "10GbE",
            Interconnect::Ethernet1G => "1GbE",
        }
    }

    pub fn speed_factor(&self) -> f64 {
        match self {
            Interconnect::InfiniBand => 0.85,
            Interconnect::Ethernet10G => 0.60,
            Interconnect::Ethernet1G => 0.35,
        }
    }

    pub fn parse_label(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "infiniband" | "ib" | "rdma" => Some(Interconnect::InfiniBand),
            "10g" | "10gbe" | "10gb" => Some(Interconnect::Ethernet10G),
            "1g" | "1gbe" | "1gb" | "ethernet" => Some(Interconnect::Ethernet1G),
            _ => None,
        }
    }
}

/// A single node in a cluster.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterNode {
    pub name: String,
    pub ram_gb: f64,
    pub gpus: Vec<GpuInfo>,
}

impl ClusterNode {
    pub fn total_vram_gb(&self) -> f64 {
        self.gpus
            .iter()
            .filter_map(|g| g.vram_gb)
            .map(|v| v * 1.0) // each entry already accounts for count via parsing
            .sum()
    }

    pub fn total_gpu_count(&self) -> u32 {
        self.gpus.iter().map(|g| g.count).sum()
    }

    /// Parse a node spec string: `[COUNTx]RAM[,GPU_SPEC]`
    ///
    /// GPU_SPEC = `[NUMx]GPU_NAME[-VRAM]`
    ///
    /// The optional leading `COUNTx` multiplier creates COUNT identical nodes.
    ///
    /// Examples:
    ///   `64G` — 64 GB RAM, no GPU
    ///   `64G,RTX4090` — 64 GB RAM, 1x RTX 4090 (VRAM from known DB)
    ///   `64G,RTX4090-24G` — 64 GB RAM, 1x RTX 4090 with 24 GB VRAM
    ///   `128G,2xA100-80G` — 128 GB RAM, 2x A100 with 80 GB each
    ///   `64G,RTX3090-24G,A100-80G` — mixed GPUs
    ///
    /// With multiplier:
    ///   `100x64G,RTX4090-24G` — 100 identical nodes
    pub fn parse(spec: &str, node_index: usize) -> Result<(Vec<Self>, usize), String> {
        let (count, remainder) = parse_multiplier(spec);

        let parts: Vec<&str> = remainder.split(',').map(|s| s.trim()).collect();
        if parts.is_empty() {
            return Err("Empty node specification".to_string());
        }

        let ram_gb = parse_size_gb(parts[0])
            .ok_or_else(|| format!("Invalid RAM size '{}'. Expected e.g. 64G, 32000M", parts[0]))?;

        let mut gpus = Vec::new();
        for gpu_spec in &parts[1..] {
            let gpu = parse_gpu_spec(gpu_spec)?;
            gpus.push(gpu);
        }

        let mut nodes = Vec::with_capacity(count);
        for i in 0..count {
            nodes.push(ClusterNode {
                name: format!("node-{}", node_index + i),
                ram_gb,
                gpus: gpus.clone(),
            });
        }

        Ok((nodes, count))
    }
}

/// A cluster of nodes that can run distributed inference.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterSpec {
    pub nodes: Vec<ClusterNode>,
    pub interconnect: Interconnect,
}

impl ClusterSpec {
    pub fn new(nodes: Vec<ClusterNode>, interconnect: Interconnect) -> Self {
        Self {
            nodes,
            interconnect,
        }
    }

    pub fn total_ram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.ram_gb).sum()
    }

    pub fn total_vram_gb(&self) -> f64 {
        self.nodes.iter().map(|n| n.total_vram_gb()).sum()
    }

    pub fn total_gpu_count(&self) -> u32 {
        self.nodes.iter().map(|n| n.total_gpu_count()).sum()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Synthesize a virtual `SystemSpecs` representing the aggregated cluster.
    ///
    /// Distributed inference frameworks (vLLM tensor parallel, llama.cpp RPC,
    /// DeepSpeed) can split model layers across nodes, so we sum VRAM and RAM.
    /// The speed penalty from network overhead is applied separately during
    /// fit scoring via `Interconnect::speed_factor()`.
    pub fn aggregate_specs(&self) -> SystemSpecs {
        let total_ram: f64 = self.nodes.iter().map(|n| n.ram_gb).sum();
        let total_vram: f64 = self.total_vram_gb();
        let total_gpus: u32 = self.total_gpu_count();

        let all_gpus: Vec<GpuInfo> = self.nodes.iter().flat_map(|n| n.gpus.clone()).collect();

        let (primary_name, primary_backend, per_gpu_vram) = if let Some(best) = all_gpus
            .iter()
            .filter(|g| g.vram_gb.is_some())
            .max_by(|a, b| {
                a.vram_gb
                    .unwrap_or(0.0)
                    .partial_cmp(&b.vram_gb.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
            (
                best.name.clone(),
                best.backend,
                best.vram_gb.unwrap_or(0.0),
            )
        } else {
            ("None".to_string(), GpuBackend::CpuX86, 0.0)
        };

        let has_gpu = total_gpus > 0 && total_vram > 0.0;

        let total_cpu_cores: usize = self.nodes.len() * 8;

        SystemSpecs {
            total_ram_gb: total_ram,
            available_ram_gb: total_ram * 0.85,
            total_cpu_cores,
            cpu_name: format!("Cluster ({} nodes)", self.nodes.len()),
            has_gpu,
            gpu_vram_gb: if has_gpu { Some(per_gpu_vram) } else { None },
            total_gpu_vram_gb: if has_gpu { Some(total_vram) } else { None },
            gpu_name: if has_gpu {
                Some(format!("{}x {} (across {} nodes)", total_gpus, primary_name, self.nodes.len()))
            } else {
                None
            },
            gpu_count: total_gpus,
            unified_memory: false,
            backend: primary_backend,
            gpus: all_gpus,
            cluster_mode: true,
            cluster_node_count: self.nodes.len() as u32,
        }
    }

    pub fn display(&self) {
        println!("\n╔══════════════════════════════════════════════╗");
        println!("║         🖧  Cluster Configuration            ║");
        println!("╠══════════════════════════════════════════════╣");
        println!(
            "║  Nodes:         {:<29}║",
            self.node_count()
        );
        println!(
            "║  Total RAM:     {:<29}║",
            format!("{:.1} GB", self.total_ram_gb())
        );
        if self.total_vram_gb() > 0.0 {
            println!(
                "║  Total VRAM:    {:<29}║",
                format!("{:.1} GB", self.total_vram_gb())
            );
            println!(
                "║  Total GPUs:    {:<29}║",
                self.total_gpu_count()
            );
        }
        println!(
            "║  Interconnect:  {:<29}║",
            self.interconnect.label()
        );
        println!("╠══════════════════════════════════════════════╣");
        let groups = self.group_identical_nodes();
        for (node, count) in &groups {
            let gpu_desc = if node.gpus.is_empty() {
                "no GPU".to_string()
            } else {
                node.gpus
                    .iter()
                    .map(|g| {
                        let vram = g
                            .vram_gb
                            .map(|v| format!(" ({:.0}G)", v))
                            .unwrap_or_default();
                        if g.count > 1 {
                            format!("{}x{}{}", g.count, g.name, vram)
                        } else {
                            format!("{}{}", g.name, vram)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let prefix = if *count > 1 {
                format!("{}x", count)
            } else {
                String::new()
            };
            println!(
                "║  {}{:.0}G RAM, {:<30}║",
                prefix, node.ram_gb, gpu_desc
            );
        }
        println!("╚══════════════════════════════════════════════╝");
    }

    /// Group identical nodes for compact display.
    fn group_identical_nodes(&self) -> Vec<(&ClusterNode, usize)> {
        let mut groups: Vec<(&ClusterNode, usize)> = Vec::new();
        for node in &self.nodes {
            let found = groups.iter_mut().find(|(representative, _)| {
                (representative.ram_gb - node.ram_gb).abs() < 0.01
                    && representative.gpus.len() == node.gpus.len()
                    && representative.gpus.iter().zip(node.gpus.iter()).all(|(a, b)| {
                        a.name == b.name && a.count == b.count && a.vram_gb == b.vram_gb
                    })
            });
            if let Some((_, count)) = found {
                *count += 1;
            } else {
                groups.push((node, 1));
            }
        }
        groups
    }
}

// ────────────────────────────────────────────────────────────────────
// Parsing helpers
// ────────────────────────────────────────────────────────────────────

/// Parse an optional leading multiplier: `NUMx` prefix.
/// Returns (count, remainder). If no multiplier, count=1.
///
/// Distinguishes node multiplier from GPU count by requiring the multiplier
/// to appear before the RAM size (which always contains G/M/T).
/// E.g. `100x64G,RTX4090` → (100, "64G,RTX4090")
///       `64G,2xA100`     → (1, "64G,2xA100")  — no node multiplier
fn parse_multiplier(spec: &str) -> (usize, &str) {
    if let Some(pos) = spec.find('x') {
        let prefix = &spec[..pos];
        let after = &spec[pos + 1..];
        if let Ok(n) = prefix.parse::<usize>()
            && n > 0
            && !after.is_empty()
            && after.starts_with(|c: char| c.is_ascii_digit())
        {
            return (n, after);
        }
    }
    (1, spec)
}

fn parse_size_gb(s: &str) -> Option<f64> {
    let s = s.trim().to_uppercase();
    if let Some(num) = s.strip_suffix('T') {
        num.parse::<f64>().ok().map(|v| v * 1024.0)
    } else if let Some(num) = s.strip_suffix('G') {
        num.parse::<f64>().ok()
    } else if let Some(num) = s.strip_suffix('M') {
        num.parse::<f64>().ok().map(|v| v / 1024.0)
    } else {
        s.parse::<f64>().ok()
    }
}

/// Known GPU VRAM database (common models).
fn known_gpu_vram(name: &str) -> Option<f64> {
    let n = name.to_uppercase();
    // NVIDIA
    if n.contains("H100") { return Some(80.0); }
    if n.contains("H200") { return Some(141.0); }
    if n.contains("A100") && n.contains("80") { return Some(80.0); }
    if n.contains("A100") && n.contains("40") { return Some(40.0); }
    if n.contains("A100") { return Some(80.0); }
    if n.contains("A10G") { return Some(24.0); }
    if n.contains("A6000") { return Some(48.0); }
    if n.contains("A5000") { return Some(24.0); }
    if n.contains("A4000") { return Some(16.0); }
    if n.contains("L40S") { return Some(48.0); }
    if n.contains("L40") { return Some(48.0); }
    if n.contains("L4") { return Some(24.0); }
    if n.contains("RTX4090") || n.contains("4090") { return Some(24.0); }
    if n.contains("RTX4080") || n.contains("4080") { return Some(16.0); }
    if n.contains("RTX4070") || n.contains("4070") { return Some(12.0); }
    if n.contains("RTX3090") || n.contains("3090") { return Some(24.0); }
    if n.contains("RTX3080") || n.contains("3080") { return Some(10.0); }
    if n.contains("RTX3070") || n.contains("3070") { return Some(8.0); }
    if n.contains("V100") && n.contains("32") { return Some(32.0); }
    if n.contains("V100") { return Some(16.0); }
    if n.contains("T4") { return Some(16.0); }
    if n.contains("P100") { return Some(16.0); }
    if n.contains("P40") { return Some(24.0); }
    if n.contains("RTX5090") || n.contains("5090") { return Some(32.0); }
    if n.contains("RTX5080") || n.contains("5080") { return Some(16.0); }
    if n.contains("B200") { return Some(192.0); }
    if n.contains("B100") { return Some(192.0); }
    if n.contains("GB200") { return Some(192.0); }
    // AMD
    if n.contains("MI300X") { return Some(192.0); }
    if n.contains("MI300") { return Some(128.0); }
    if n.contains("MI250") { return Some(128.0); }
    if n.contains("MI210") { return Some(64.0); }
    if n.contains("MI100") { return Some(32.0); }
    if n.contains("7900XTX") { return Some(24.0); }
    if n.contains("7900XT") { return Some(20.0); }
    if n.contains("W7900") { return Some(48.0); }
    None
}

/// Parse a GPU spec: `[NUMx]NAME[-VRAM]`
///
/// Examples: `RTX4090`, `RTX4090-24G`, `2xA100-80G`, `MI300X`
fn parse_gpu_spec(spec: &str) -> Result<GpuInfo, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("Empty GPU specification".to_string());
    }

    let (count, rest) = if let Some(idx) = spec.find('x') {
        let count_str = &spec[..idx];
        if let Ok(c) = count_str.parse::<u32>() {
            (c, &spec[idx + 1..])
        } else {
            (1, spec)
        }
    } else {
        (1, spec)
    };

    let (name, vram_gb) = if let Some(last_dash) = rest.rfind('-') {
        let maybe_vram = &rest[last_dash + 1..];
        if let Some(gb) = parse_size_gb(maybe_vram) {
            (rest[..last_dash].to_string(), Some(gb))
        } else {
            (rest.to_string(), None)
        }
    } else {
        (rest.to_string(), None)
    };

    let vram_gb = vram_gb.or_else(|| known_gpu_vram(&name));

    let backend = infer_backend(&name);

    Ok(GpuInfo {
        name,
        vram_gb: vram_gb.map(|v| v * count as f64),
        backend,
        count,
        unified_memory: false,
    })
}

fn infer_backend(name: &str) -> GpuBackend {
    let n = name.to_uppercase();
    if (n.contains("MI") && (n.contains("300") || n.contains("250") || n.contains("210") || n.contains("100")))
        || n.contains("7900")
        || n.contains("W7900")
    {
        GpuBackend::Rocm
    } else {
        GpuBackend::Cuda
    }
}

# Cluster Mode - Implementation Plan

## Goal
Allow users to define a cluster of machines and evaluate which LLM models
can run across the combined resources using distributed inference (e.g.,
tensor parallelism via vLLM, llama.cpp RPC, etc.).

## Design Decisions

### How users define a cluster
- **CLI**: `llmfit cluster --node "64G,RTX4090" --node "32G,2xA100-80G" --node "128G"`
  Each `--node` specifies: `RAM[,GPU_SPEC]` where GPU_SPEC = `[NUMx]GPU_NAME[-VRAM]`
- **Config file**: `~/.config/llmfit/cluster.toml` for saved cluster definitions
- **TUI**: Future - cluster tab/view (Phase 2, not this PR)

### How cluster resources aggregate
- **Total VRAM**: Sum of all GPU VRAM across all nodes (tensor parallelism)
- **Total RAM**: Sum of all node RAM (for CPU offload fallback)
- **Interconnect penalty**: Speed estimation gets a penalty factor for
  cross-node communication (user can specify: local, ethernet, infiniband)
- **Backend**: Use the most common GPU backend across nodes; mixed backends
  get a penalty

### Fit analysis for cluster
- Reuse `ModelFit::analyze()` with a synthesized "virtual" `SystemSpecs`
  that represents aggregated cluster resources
- Add `RunMode::Distributed` variant for models that need multiple nodes
- Speed penalty based on interconnect type

## Phases

### Phase 1: Core cluster support (this PR)
- [x] 1. Add `ClusterNode` and `ClusterSpec` structs to `llmfit-core/src/cluster.rs`
- [x] 2. Add `ClusterSpec::aggregate_specs()` -> `SystemSpecs` (virtual combined system)
- [x] 3. Add `RunMode::Distributed` variant
- [x] 4. Add `cluster` CLI subcommand to parse `--node` args
- [x] 5. Add `display_cluster_fits()` to display.rs
- [x] 6. Wire up in main.rs
- [x] 7. Add `COUNTx` multiplier prefix for identical nodes (e.g. `100x64G,RTX4090`)

### Phase 2: Enhanced cluster (future)
- [ ] Cluster definition via TOML config file
- [ ] TUI cluster view/tab
- [ ] Per-node fit breakdown (which layers on which node)
- [ ] Network bandwidth estimation

## Status: Phase 1 COMPLETE

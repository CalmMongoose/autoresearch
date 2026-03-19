# UFR Autoresearch Program

## Goal
Explore architectures that optimize for both **performance** (low val_bpb) and **structural interpretability** (factorization, modularity, specialization). We're seeking Unifying Factored Representations (UFR) where the model's computational graph resembles a concept graph or entity relationship diagram rather than an entangled high-dimensional structure.

## Composite Objective
The primary metric is `composite_score` which combines:
- **50%** Performance (val_bpb improvement)
- **15%** Head consistency (lower = heads more distinct/specialized)
- **10%** Activation entropy (higher = more diverse representations)
- **10%** Effective rank (higher = more capacity utilized)
- **10%** Weight sparsity (lower = more weights actively used)
- **5%** Head entropy (higher = more focused attention patterns)

**Key insight**: A good architecture scores well on BOTH performance AND structural properties.

## What to Explore

### Architecture Modifications
1. **Attention head specialization**: Find patterns where different heads learn distinct functions (positional, semantic, syntactic)
2. **Layer-wise specialization**: Can early/late layers develop clear roles?
3. **Sparse connectivity**: Which connections can be removed without hurting performance?
4. **Modular substructures**: Can the model be organized into interpretable components?

### Search Strategies
1. **Start simple**: Establish baseline with default config
2. **Factorization pressure**: Try reducing head count, increasing per-head dimension
3. **Modularity experiments**: Add gating mechanisms, try mixture-of-experts patterns
4. **Ablation studies**: Remove components and observe what happens

### Constraints
- Keep val_bpb within 10% of baseline (don't sacrifice all performance for structure)
- Maintain training stability (no NaN, no explosions)
- Single GPU only (H100)
- 5-minute time budget per run

## Output Files
- `metrics_output.json` - Detailed metrics for each run
- Git commits should include brief description of architectural change

## Success Criteria
1. Find architectures with composite_score > 1.2x baseline
2. Identify patterns that achieve specialization without performance loss
3. Document which structural changes reliably improve interpretability metrics

Remember: We're not just optimizing for speed/loss - we're exploring the Pareto frontier of capability vs interpretability.

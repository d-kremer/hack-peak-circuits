# Classical Simulation of Peaked Quantum Circuits

Code for classically simulating the peaked quantum circuits described in [Gharibyan et al. (arXiv:2510.25838)](https://arxiv.org/abs/2510.25838). This repository contains the methods used in our paper *Efficient Classical Simulation of Heuristic Peaked Quantum Circuits*.

The peaked circuit QASM files are available from the [BlueQubit Peak Portal](https://app.bluequbit.io/hackathons/oEOtLSSrPSVH60Ah?tab=problems) and the [Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io).

## Methods

Three simulation approaches are included, each targeting different circuit types and sizes:

### Low-Bond MPS with Bitstring Distillation

Runs a low-bond-dimension MPS simulation that does not resolve the peak directly, then distills the peak bitstring from samples via majority voting on individual bit probabilities. Useful as a fast heuristic for circuits where exact contraction is not needed.

- **Notebook**: `peaked-circuit-distillation.ipynb`
- **Target**: Moderate-depth circuits (e.g., `P4_golden_mountain`, 48 qubits, 5096 CZ gates)
- **Hardware**: Single GPU, ~6 minutes

### Tensor Network Operator (TNO) Contraction

A TNO-based contraction. Iteratively absorbs circuit layers from both sides of the midpoint and compresses using tensor network methods. Works well for circuits that have no hidden permutations.

- **Notebooks**: `peaked-circuit-tno.ipynb` (GPU), `peaked-circuit-tno-cpu.ipynb` (CPU)
- **Target**: Peak circuits without permutations (e.g., `peaked_circuit_heavy_hex_49x4020`, `peaked_circuit_heavy_hex_49x5072`).
- **Hardware**: Single GPU or CPU, seconds to minutes


### MPO Iterative Cancellation with Unswapping

Splits the circuit at the midpoint and transpiles it to linear connectivity. Then contracts both halves into a central Matrix Product Operator (MPO), and uses a greedy "unswapping" procedure to extract hidden permutation structure and keep the MPO size bounded. Produces near-exact samples from the circuit output distribution.

- **Notebook**: `peaked-circuit-unswapping.ipynb`
- **Target**: Large all-to-all circuits (e.g., `peaked_circuit_P9_Hqap_56x1917`, 56 qubits, 1917 gates) with hidden permutations.
- **Hardware**: Single GPU (Nvidia A100 80 GB), ~1 hour runtime


## Code Structure

| File | Description |
|------|-------------|
| `unswap.py` | MPO iterative cancellation with unswapping, rewiring, and adaptive side selection |
| `circuit_mpo.py` | MPO construction from circuits, MPO-MPO composition, and swap application |
| `utils.py` | Circuit layering utilities, TNO contraction, MPS sampling, and bitstring extraction |


## Citation

If you use this code, please cite:

```bibtex
@article{kremer2026peaked,
    title   = {Efficient Classical Simulation of Heuristic Peaked Quantum Circuits},
    author  = {Kremer, David and Dupuis, Nicolas},
    year    = {2026},
    eprint  = {2604.21908},
    archivePrefix = {arXiv},
    primaryClass  = {quant-ph},
    url     = {https://arxiv.org/abs/2604.21908}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

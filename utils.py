import numpy as np
from qiskit import QuantumCircuit

import quimb.tensor as qtn
from quimb.tensor.tensor_arbgeom_compress import tensor_network_ag_compress

from qiskit_quimb import quimb_circuit
import torch

from qiskit.converters import circuit_to_dag, dag_to_circuit


def iter_layers(qc):
    for layer in circuit_to_dag(qc).layers():
        yield dag_to_circuit(layer['graph'])


DEVICE = "cuda:0"
def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device=DEVICE)

# method: 'l2bp', 'local-early', 'local-late', 'projector', 'superorthogonal'
def contract_core(layered_circuit, chunk_size=4, method='local-late', max_bond=32, cutoff=0.1, equalize_norms=True):
    N = layered_circuit[0].num_qubits
    L = len(layered_circuit)
    M = L // 2

    # Layers before and after the center
    layers_first_qc = layered_circuit[:M][::-1]
    layers_last_qc = layered_circuit[-M:]

    # This is a placeholder identity circuit to have all the sites
    qc_id = QuantumCircuit(N)
    qc_id.x(list(range(N)))
    qc_id.x(list(range(N)))

    # Start the TNO at the center of the circuit
    tno_core = quimb_circuit(layered_circuit[M].compose(qc_id), to_backend=to_backend).get_uni()

    # Merge and contract chunks of layers to the left and right of the circuit
    for i in range(0, M+1, chunk_size):

        # Make the left and right chunks
        qc_left = qc_id.copy()
        for j in reversed(range(chunk_size)):
            if (i+j) >= len(layers_first_qc):
                break
            qc_left = qc_left.compose(layers_first_qc[i+j])

        qc_right = qc_id.copy()
        for j in range(chunk_size):
            if (i+j) >= len(layers_last_qc):
                break
            qc_right = qc_right.compose(layers_last_qc[i+j])

        # Merge them with the current core
        tn_left = quimb_circuit(qc_left, to_backend=to_backend).get_uni()
        tn_right = quimb_circuit(qc_right, to_backend=to_backend).get_uni()
        tno_core = tn_left.gate_upper_with_op_lazy(tno_core).gate_upper_with_op_lazy(tn_right)

        # Compress the TNO
        tno_core = tensor_network_ag_compress(
            tno_core,
            method=method, 
            cutoff=cutoff,
            max_bond=max_bond,
            site_tags=tno_core.site_tags,
            canonize=True,
            equalize_norms=equalize_norms,
        )

        tno_core = tno_core.squeeze()
        shapes_flat = [s for t in tno_core for s in t.shape]
        shapes_count = [len(t.shape) for t in tno_core]
        elem_counts = [np.prod(t.shape).item() for t in tno_core]

        print(f"    (|{M-1-(i+j)}-{M-1-i}|-{M}-|{M+1+i}-{M+1+i+j}|)/({len(layered_circuit)}) -> max_bond = {tno_core.max_bond()}, max_links = {max(shapes_count)}, total_elems = {sum(elem_counts)}, total_shapes = {np.sum(shapes_flat).item()}")

    return tno_core


def tno_to_tne(tno, max_bond=8, cutoff=0.01):
    nq = len(tno.sites)
    tne = quimb_circuit(QuantumCircuit(nq), to_backend=to_backend).psi

    # Evolve state by op
    tne = qtn.tensor_arbgeom.tensor_network_apply_op_vec(tno, tne)

    # Compress
    tne = tensor_network_ag_compress(
        tne,
        method='local-late', 
        cutoff=cutoff,
        max_bond=max_bond,
        site_tags=tne.site_tags,
        canonize=True,
        #lazy=lazy,
        equalize_norms=True,
        #progbar=True,
        #to_backend=to_backend,
    )

    tne = tne.squeeze()
    shapes_count = [len(t.shape) for t in tne]
    print(f"max_bond = {tne.max_bond()}, max_links = {max(shapes_count)}")

    return tne


def extract_bitstring(tne):
    # Predict bitstring with marginals
    nq = len(tne.sites)
    pred_bs = ''
    p0s = []
    for ii in range(nq):
        Pi0 = torch.tensor(np.array([[1., 0.],[0., 0.]]), device=DEVICE, dtype=torch.cfloat)
        p0 = tne.local_expectation(Pi0, where=[ii], max_bond=2, optimize="auto", normalized=True).real.item()
        p0s.append(p0)
        pred_bs += '1' if p0 < 0.5 else '0'
        #print(f"({ii}) -> {p0:.3f} | {pred_bs}")
    return pred_bs, p0s
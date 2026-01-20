"""
Tests for circuit utility functions.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from pyrkm.circuit_utils import Circuit


@pytest.fixture
def circuit():
    graph = nx.Graph()
    graph.add_nodes_from(
        [(0, {"pos": (0, 0)}), (1, {"pos": (1, 0)}), (2, {"pos": (0, 1)}), (3, {"pos": (1, 1)})]
    )
    graph.add_edges_from([(0, 1), (1, 3), (3, 2), (2, 0)])
    return Circuit(graph)


def test_set_conductances(circuit) -> None:
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    np.testing.assert_array_equal(circuit.conductances, conductances)


def test_hessian(circuit) -> None:
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    hessian = circuit._hessian()
    assert hessian.shape == (4, 4)


def test_constraint_matrix(circuit) -> None:
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    assert Q.shape == (4, 2)


@pytest.mark.skip(reason="Test is currently broken")
def test_extended_hessian(circuit) -> None:
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    extended_hessian = circuit._extended_hessian(Q)
    assert extended_hessian.shape == (6, 6)


def test_solve(circuit) -> None:
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    f = np.array([1.0, -1.0])
    V = circuit.solve(Q, f)
    assert len(V) == 4


def test_circuit_initialization(circuit) -> None:
    """
    Test circuit is properly initialized.
    """
    assert circuit.graph is not None
    assert circuit.graph.number_of_nodes() == 4
    assert circuit.graph.number_of_edges() == 4


def test_conductances_array_type(circuit) -> None:
    """
    Test conductances are stored correctly.
    """
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    assert isinstance(circuit.conductances, (list, np.ndarray))
    assert len(circuit.conductances) == 4


def test_hessian_symmetry(circuit) -> None:
    """
    Test hessian matrix is symmetric.
    """
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    hessian = circuit._hessian()
    # Convert to dense if sparse
    if hasattr(hessian, "toarray"):
        hessian_dense = hessian.toarray()
    else:
        hessian_dense = hessian
    assert np.allclose(hessian_dense, hessian_dense.T)


def test_hessian_diagonal_sum(circuit) -> None:
    """
    Test hessian diagonal properties.
    """
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    hessian = circuit._hessian()
    # Convert to dense if sparse
    if hasattr(hessian, "toarray"):
        hessian_dense = hessian.toarray()
    else:
        hessian_dense = hessian
    # Each row sum should be close to zero (Kirchhoff's law)
    row_sums = np.sum(hessian_dense, axis=1)
    assert np.allclose(row_sums, 0, atol=1e-10)


def test_constraint_matrix_shape(circuit) -> None:
    """
    Test constraint matrix has correct shape.
    """
    indices_nodes = np.array([0, 2, 3])
    Q = circuit.constraint_matrix(indices_nodes)
    assert Q.shape == (4, 3)


def test_constraint_matrix_single_node(circuit) -> None:
    """
    Test constraint matrix with single node.
    """
    indices_nodes = np.array([1])
    Q = circuit.constraint_matrix(indices_nodes)
    assert Q.shape == (4, 1)


def test_solve_with_different_sources(circuit) -> None:
    """
    Test solving with different current sources.
    """
    conductances = [1.0, 1.0, 1.0, 1.0]
    circuit.setConductances(conductances)
    indices_nodes = np.array([0, 3])
    Q = circuit.constraint_matrix(indices_nodes)

    # Different current configurations
    f1 = np.array([1.0, -1.0])
    V1 = circuit.solve(Q, f1)
    assert len(V1) == 4

    f2 = np.array([2.0, -2.0])
    V2 = circuit.solve(Q, f2)
    assert len(V2) == 4
    # V2 should be approximately twice V1 (linear system)
    assert np.allclose(V2, 2 * V1, rtol=1e-5)


def test_zero_conductances_handling(circuit) -> None:
    """
    Test handling of zero conductances.
    """
    conductances = [0.0, 1.0, 2.0, 3.0]
    circuit.setConductances(conductances)
    hessian = circuit._hessian()
    assert hessian is not None
    # Should not raise errors


def test_large_conductances(circuit) -> None:
    """
    Test circuit with large conductances.
    """
    conductances = [100.0, 200.0, 150.0, 250.0]
    circuit.setConductances(conductances)
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    f = np.array([1.0, -1.0])
    V = circuit.solve(Q, f)
    assert len(V) == 4
    assert not np.any(np.isnan(V))
    assert not np.any(np.isinf(V))

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from pyrkm.circuit_utils import Circuit


@pytest.fixture
def circuit():
    graph = nx.Graph()
    graph.add_nodes_from([(0, {
        'pos': (0, 0)
    }), (1, {
        'pos': (1, 0)
    }), (2, {
        'pos': (0, 1)
    }), (3, {
        'pos': (1, 1)
    })])
    graph.add_edges_from([(0, 1), (1, 3), (3, 2), (2, 0)])
    return Circuit(graph)


def test_set_conductances(circuit):
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    np.testing.assert_array_equal(circuit.conductances, conductances)


def test_hessian(circuit):
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    hessian = circuit._hessian()
    assert hessian.shape == (4, 4)


def test_constraint_matrix(circuit):
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    assert Q.shape == (4, 2)


@pytest.mark.skip(reason='Test is currently broken')
def test_extended_hessian(circuit):
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    extended_hessian = circuit._extended_hessian(Q)
    assert extended_hessian.shape == (6, 6)


def test_solve(circuit):
    conductances = [1.0, 2.0, 3.0, 4.0]
    circuit.setConductances(conductances)
    indices_nodes = np.array([0, 2])
    Q = circuit.constraint_matrix(indices_nodes)
    f = np.array([1.0, -1.0])
    V = circuit.solve(Q, f)
    assert len(V) == 4

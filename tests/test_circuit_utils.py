from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from pyrkm.circuit_utils import Circuit


class TestCircuit(unittest.TestCase):

    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(0, {
            'pos': (0, 0)
        }), (1, {
            'pos': (1, 0)
        }), (2, {
            'pos': (0, 1)
        }), (3, {
            'pos': (1, 1)
        })])
        self.graph.add_edges_from([(0, 1), (1, 3), (3, 2), (2, 0)])
        self.circuit = Circuit(self.graph)

    def test_set_conductances(self):
        conductances = [1.0, 2.0, 3.0, 4.0]
        self.circuit.setConductances(conductances)
        np.testing.assert_array_equal(self.circuit.conductances, conductances)

    def test_hessian(self):
        conductances = [1.0, 2.0, 3.0, 4.0]
        self.circuit.setConductances(conductances)
        hessian = self.circuit._hessian()
        self.assertEqual(hessian.shape, (4, 4))

    def test_constraint_matrix(self):
        indices_nodes = np.array([0, 2])
        Q = self.circuit.constraint_matrix(indices_nodes)
        self.assertEqual(Q.shape, (4, 2))

    def test_extended_hessian(self):
        indices_nodes = np.array([0, 2])
        Q = self.circuit.constraint_matrix(indices_nodes)
        extended_hessian = self.circuit._extended_hessian(Q)
        self.assertEqual(extended_hessian.shape, (6, 6))

    def test_solve(self):
        conductances = [1.0, 2.0, 3.0, 4.0]
        self.circuit.setConductances(conductances)
        indices_nodes = np.array([0, 2])
        Q = self.circuit.constraint_matrix(indices_nodes)
        f = np.array([1.0, -1.0])
        V = self.circuit.solve(Q, f)
        self.assertEqual(len(V), 4)


if __name__ == '__main__':
    unittest.main()

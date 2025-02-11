from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve


class Circuit(object):
    ''' Class to simulate a circuit with trainable conductances

    Parameters
    ----------
    graph : str or networkx.Graph
        If str, it is the path to the file containing the graph. If networkx.Graph, it is the graph itself.

    Attributes
    ----------
    graph : networkx.Graph
        Graph specifying the nodes and edges in the network.
        A conductance parameter is associated with each edge.
        A trainable edge will be updated during training.
    n : int
        Number of nodes in the graph.
    ne : int
        Number of edges in the graph.
    pts : numpy.ndarray
        Positions of the nodes in the graph.
    '''

    def __init__(self, graph):
        if isinstance(graph, str):
            self.graph = nx.read_gpickle(graph)
        else:
            self.graph = graph

        self.n = len(self.graph.nodes)
        self.ne = len(self.graph.edges)
        self.pts = np.array(
            [self.graph.nodes[node]['pos'] for node in graph.nodes])
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)

    def setConductances(self, conductances):
        ''' Set the conductances of the edges in the graph.

        Parameters
        ----------
        conductances : list of float or numpy.ndarray
            Conductances of the edges. Must have the same length as the number of edges.
        '''
        assert len(
            conductances
        ) == self.ne, 'conductances must have the same length as the number of edges'
        if isinstance(conductances, list):
            conductances = np.array(conductances)
        self.conductances = conductances

    def _hessian(self):
        ''' Compute the Hessian of the network with respect to the conductances.

        Returns
        -------
        scipy.sparse.csr_matrix
            Hessian matrix.
        '''
        return (self.incidence_matrix * self.conductances).dot(
            self.incidence_matrix.T)

    def constraint_matrix(self, indices_nodes):
        ''' Compute the constraint matrix Q for the circuit and the nodes represented by indices_nodes.

        Parameters
        ----------
        indices_nodes : numpy.ndarray
            Array with the indices of the nodes to be constrained.
            The nodes themselves are given by
            np.array(self.graph.nodes)[indices_nodes].

        Returns
        -------
        scipy.sparse.csr_matrix
            Constraint matrix Q: a sparse constraint rectangular
            matrix of size n x len(indices_nodes).
            Its entries are only 1 or 0.
            Q.Q^T is a projector onto to the space of the nodes.
        '''
        if len(indices_nodes) == 0:
            raise ValueError('indicesNodes must be a non-empty array.')
        Q = csr_matrix((np.ones(len(indices_nodes)),
                        (indices_nodes, np.arange(len(indices_nodes)))),
                       shape=(self.n, len(indices_nodes)))
        return Q

    def _extended_hessian(self, Q):
        ''' Extend the hessian of the network with the constraint matrix Q.

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q

        Returns
        -------
        scipy.sparse.csr_matrix
            Extended Hessian. H is a sparse matrix of size
            (n + len(indices_nodes)) x (n + len(indices_nodes)).
        '''
        sparseExtendedHessian = bmat([[self._hessian(), Q], [Q.T, None]],
                                     format='csr',
                                     dtype=float)
        return sparseExtendedHessian

    def solve(self, Q, f):
        ''' Solve the circuit with the constraint matrix Q and the source vector f.

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q
        f : numpy.ndarray
            Source vector f. f has size len(indices_nodes).

        Returns
        -------
        numpy.ndarray
            Solution vector V. V has size n.
        '''
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        if len(f) != Q.shape[1]:
            raise ValueError('Source vector f has the wrong size.')
        H = self._extended_hessian(Q)
        f_extended = np.hstack([np.zeros(self.n), f])
        V = spsolve(H, f_extended)[:self.n]
        return V

    def plot_node_state(self,
                        node_state,
                        title=None,
                        lw=0.5,
                        cmap='RdYlBu_r',
                        size_factor=100):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        node_state : numpy.ndarray
            State of the nodes in the graph. node_state has size n.
        title : str, optional
            Title of the plot.
        lw : float, optional
            Line width of the edges.
        cmap : str, optional
            Colormap to use for the plot.
        size_factor : float, optional
            Factor to scale the size of the nodes.
        '''
        posX = self.pts[:, 0]
        posY = self.pts[:, 1]
        norm = plt.Normalize(vmin=np.min(node_state), vmax=np.max(node_state))
        fig, axs = plt.subplots(1,
                                1,
                                figsize=(4, 4),
                                constrained_layout=True,
                                sharey=True)
        axs.scatter(posX,
                    posY,
                    s=size_factor * np.abs(node_state[:]),
                    c=node_state[:],
                    edgecolors='black',
                    linewidth=lw,
                    cmap=cmap,
                    norm=norm)
        axs.set(aspect='equal')
        axs.set_xticks([])
        axs.set_yticks([])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axs,
                     shrink=0.5)
        axs.set_title(title)

    def plot_edge_state(self, edge_state, title=None, lw=0.5, cmap='RdYlBu_r'):
        ''' Plot the state of the edges in the graph.

        Parameters
        ----------
        edge_state : numpy.ndarray
            State of the edges in the graph. edge_state has size ne.
        title : str, optional
            Title of the plot.
        lw : float, optional
            Line width of the edges.
        cmap : str, optional
            Colormap to use for the plot.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        pos_edges = np.array([
            np.array([
                self.graph.nodes[edge[0]]['pos'],
                self.graph.nodes[edge[1]]['pos']
            ]).T for edge in self.graph.edges()
        ])
        norm = plt.Normalize(vmin=np.min(edge_state), vmax=np.max(edge_state))
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i, 0],
                     pos_edges[i, 1],
                     color=_cmap(norm(edge_state[i])))
        axs.set_xticks([])
        axs.set_yticks([])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=axs,
                     shrink=0.5)
        axs.set_title(title)

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Circuit(object):
    ''' Class to simulate a circuit with trainable conductances 
    
    Parameters
    ----------
    graph   :   str or networkx.Graph
                If str, it is the path to the file containing the graph. If networkx.Graph, it is the graph itself.
    
    Attributes
    ----------
    graph   :   networkx.Graph
                Graph specifying the nodes and edges in the network. A conductance parameter is associated with each edge. A trainable edge will
		        be updated during training.
    n : int
        Number of nodes in the graph.
    ne : int
        Number of edges in the graph.
    pts: numpy.ndarray
        Positions of the nodes in the graph.
    '''

    def __init__(self, graph):
        if type(graph) == str:
            self.graph = nx.read_gpickle(graph)
        else:
            self.graph = graph

        self.n = len(self.graph.nodes)
        self.ne = len(self.graph.edges)
        self.pts = np.array([self.graph.nodes[node]['pos'] for node in graph.nodes])
        
        self.incidence_matrix = nx.incidence_matrix(self.graph, oriented=True)

    # def setPositions(self, positions):
    #     # positions is a list of tuples
    #     assert len(positions) == self.n, 'positions must have the same length as the number of nodes'
    #     self.pts = positions

    def setConductances(self, conductances):
        # conductances is a list of floats
        assert len(conductances) == self.ne, 'conductances must have the same length as the number of edges'
        # if list, convert to numpy array
        if type(conductances) == list:
            conductances = np.array(conductances)
        self.conductances = conductances

    def _hessian(self):
        ''' Compute the Hessian of the network with respect to the conductances. '''
        return (self.incidence_matrix*self.conductances).dot(self.incidence_matrix.T)
    
    def constraint_matrix(self, indices_nodes):
        ''' Compute the constraint matrix Q for the circuit and the nodes represented by indices_nodes. 
        Q is a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
        Q.Q^T is a projector onto to the space of the nodes.

        Parameters
		----------
		indices_nodes : np.array
			Array with the indices of the nodes to be constrained. The nodes themselves are given by np.array(self.graph.nodes)[indices_nodes].

		Returns
		-------
		Q: scipy.sparse.csr_matrix
			Constraint matrix Q: a sparse constraint rectangular matrix of size n x len(indices_nodes). Its entries are only 1 or 0.
            Q.Q^T is a projector onto to the space of the nodes.
		
        '''
         # Check indicesNodes is a non-empty array
        if len(indices_nodes) == 0:
            raise ValueError('indicesNodes must be a non-empty array.')
        # Create the sparse rectangular constraint matrix Q using csr_matrix. Q has entries 1 at the indicesNodes[i] row and i column.
        Q = csr_matrix((np.ones(len(indices_nodes)), (indices_nodes, np.arange(len(indices_nodes)))), shape=(self.n, len(indices_nodes)))
        return Q
    
    def _extended_hessian(self, Q):
        ''' Extend the hessian of the network with the constraint matrix Q. 

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q

        Returns
        -------
        H : scipy.sparse.csr_matrix
            Extended Hessian. H is a sparse matrix of size (n + len(indices_nodes)) x (n + len(indices_nodes)).
        
        '''
        sparseExtendedHessian = bmat([[self._hessian(), Q], [Q.T, None]], format='csr', dtype=float)
        return sparseExtendedHessian


    
    '''
	*****************************************************************************************************
	*****************************************************************************************************

										NUMERICAL INTEGRATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

    def solve(self, Q, f):
        ''' Solve the circuit with the constraint matrix Q and the source vector f.

        Parameters
        ----------
        Q : scipy.sparse.csr_matrix
            Constraint matrix Q
        f : np.array
            Source vector f. f has size len(indices_nodes).

        Returns
        -------
        x : np.array
            Solution vector V. V has size n.
        '''
        # check that the conductances have been set
        try:
            self.conductances
        except AttributeError:
            raise AttributeError('Conductances have not been set yet.')
        # check that the source vector has the right size
        if len(f) != Q.shape[1]:
            raise ValueError('Source vector f has the wrong size.')
        # extend the hessian
        H = self._extended_hessian(Q)
        # extend f with n zeros
        f_extended = np.hstack([np.zeros(self.n), f])
        # solve the system
        V = spsolve(H, f_extended)[:self.n]
        return V






    '''
	*****************************************************************************************************
	*****************************************************************************************************

										PLOTTING AND ANIMATION

	*****************************************************************************************************
	*****************************************************************************************************
    '''

    def plot_node_state(self, node_state, title = None, lw = 0.5, cmap = 'RdYlBu_r', size_factor = 100):
        ''' Plot the state of the nodes in the graph.

        Parameters
        ----------
        node_state : np.array
            State of the nodes in the graph. node_state has size n.
        '''
        posX = self.pts[:,0]
        posY = self.pts[:,1]
        norm = plt.Normalize(vmin=np.min(node_state), vmax=np.max(node_state))
        fig, axs = plt.subplots(1,1, figsize = (4,4), constrained_layout=True,sharey=True)
        axs.scatter(posX, posY, s = size_factor*np.abs(node_state[:]), c = node_state[:],edgecolors = 'black',linewidth = lw,  cmap = cmap, norm = norm)
        axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # show the colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)

    def plot_edge_state(self, edge_state, title = None,lw = 0.5, cmap = 'RdYlBu_r'):
        ''' Plot the state of the edges in the graph.

        Parameters
        ----------
        edge_state : np.array
            State of the edges in the graph. edge_state has size ne.
        '''
        _cmap = plt.cm.get_cmap(cmap)
        pos_edges = np.array([np.array([self.graph.nodes[edge[0]]['pos'], self.graph.nodes[edge[1]]['pos']]).T for edge in self.graph.edges()])
        norm = plt.Normalize(vmin=np.min(edge_state), vmax=np.max(edge_state))
        fig, axs = plt.subplots(1,1, figsize = (4,4))
        for i in range(len(pos_edges)):
            axs.plot(pos_edges[i,0], pos_edges[i,1], color = _cmap(norm(edge_state[i])))
        # axs.set( aspect='equal')
        # remove ticks
        axs.set_xticks([])
        axs.set_yticks([])
        # show the colorbar
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs, shrink=0.5)
        # set the title of each subplot to be the corresponding eigenvalue in scientific notation
        axs.set_title(title)

 
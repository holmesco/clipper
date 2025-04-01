import sys
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
import unittest

from src.clipper.clipper import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points

class TestClipper(unittest.TestCase):
    def __init__(self, test_prob = "three-clique"):
        self.test_prob = test_prob
        if test_prob == "bunny":
            self.init_bunny_test()
        elif test_prob == "two-clique":
            self.two_clique_test()
        elif test_prob == "three-clique":
            self.three_clique_test()
        
        # Define problem
        self.prob = ConsistencyGraphProb(self.affinity)
        self.size = self.affinity.shape[0]

    def two_clique_test(self):
        """Simple test with two separate cliques"""
        self.affinity = sp.linalg.block_diag(np.ones((5,5)), np.ones((3,3)))
        # self.affinity = 
        self.x = np.vstack([np.ones((5,1)), np.zeros((3,1))])
        self.X = self.x @ self.x.T
        
    def three_clique_test(self):
        """Simple test with three overlapping cliques"""
        self.affinity = sp.linalg.block_diag(np.ones((5,5)), np.ones((3,3)))
        self.affinity[4,5] = 1
        self.affinity[5,4] = 1
        
        # self.affinity = 
        self.x = np.vstack([np.ones((5,1)), np.zeros((3,1))])
        self.X = self.x @ self.x.T


    def init_bunny_test(self):
        # Set up common variables for tests
        self.m = 100
        self.n1 = 100
        self.n2o = 10
        self.outrat = 0.9
        self.sigma = 0.01
        self.pcfile = 'examples/data/bun10k.ply'
        self.T_21 = np.eye(4)
        self.T_21[0:3, 0:3] = Rotation.random().as_matrix()
        self.T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))
        # Generate dataset
        self.D1, self.D2, self.Agt, self.A = generate_bunny_dataset(
            self.pcfile, self.m, self.n1, self.n2o, self.outrat, self.sigma, self.T_21
        )     
        # Generate affinity
        self.affinity, self.clipper = get_affinity_from_points(self.D1, self.D2, self.A) 
         # Generate a solution vector
        x = np.zeros((self.A.shape[0], 1))
        for i,a in enumerate( self.A):
            # Find any associations that are in the GT set
            if np.any(np.sum(a == self.Agt,axis=1) == 2):
                x[i] = 1
            else:
                x[i] = 0
        self.x = x
        # Generate a solution matrix
        self.X = x @ x.T

    def test_affine_constraints(self, X=None, tol = 1e-10):
        eqs, ineqs = self.prob.get_affine_constraints()
        self.assertIsInstance(eqs, list)
        self.assertIsInstance(ineqs, list)
        self.assertGreater(len(eqs) + len(ineqs), 0)
        if X is None:
            X = self.X        
        # Test Equalities
        for i, (A,b) in enumerate(eqs):
            assert np.abs(np.trace(A @ self.X) + b) < tol, ValueError(f"Equality Constraint {i} violated!")
        # Test Inequalities
        for i, (A,b) in enumerate(eqs):
            assert np.trace(A @ self.X) + b > -tol, ValueError(f"Equality Constraint {i} violated!")
          
    def test_solve_cvxpy(self):
        X, H, yvals = self.prob.solve_cvxpy(verbose=False)
        self.assertIsNotNone(X)
        self.assertIsNotNone(H)
        self.assertIsInstance(yvals, list)

    def test_solve_fusion(self):
        X, info = self.prob.solve_fusion(verbose=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        
        self.check_solution(X)
        
    
    def test_symb_fact(self, plot=False):
        self.prob.symb_fact_affinity()
        self.assertIsNotNone(self.prob.symb)    
        # test clique definitions
        if self.test_prob == "two-clique":
            # Only two cliques
            assert len(self.prob.cliques) == 2, ValueError("This test should only have two cliques")
            assert self.prob.cliques == [[0,1,2,3,4],[5,6,7]]
            mapping = {0:{0}, 1:{0},2:{0},3:{0},4:{0}, 5:{1}, 6:{1},7:{1}}
            assert self.prob.ind_to_clq == mapping, ValueError("index to clique mapping incorrect")
        elif self.test_prob == "three-clique":
            assert len(self.prob.cliques) == 3, ValueError("This test should only have two cliques")
            cliques = [set([0,1,2,3,4]),set([4,5]),set([5,6,7])]
            for clique in self.prob.cliques:
                assert set(clique) in cliques, ValueError("Cliques Incorrect")
                    
        # Test ind to clique map
        for key, clq_list in self.prob.ind_to_clq.items():
            for clq_ind in clq_list:
                assert key in self.prob.cliques[clq_ind], ValueError("index to clique map is broken")        
        # Check that fill edges are not in original sparsity pattern
        for edge_fill in self.prob.fill_edges:
            assert edge_fill not in self.prob.edges, ValueError("Fill edge is in original edge set")
        
        if plot:
            G = nx.Graph()  # Use nx.DiGraph() for a directed graph
            G.add_edges_from(self.prob.edges)
            plt.figure(figsize=(5, 5))
            pos = nx.kamada_kawai_layout(G)
            nx.draw(G, pos, with_labels=False, node_color='blue', edge_color='gray', node_size=5)
            plt.show()
                    
    def test_solve_fusion_sparse(self):
        X, info = self.prob.solve_fusion_sparse(verbose=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        self.check_solution(X)
        
    def check_solution(self,X):
        """Check that solution is rank 1 and that the solution matches what we expect."""
        # Process solution
        inliers, er, x_opt = self.prob.process_sdp_var(X)
        # Check ER
        assert er > 1e6, ValueError("Solution not rank-1")
        
        if self.test_prob == "two-clique":
            assert np.all(inliers.astype(float) == self.x.flatten()), ValueError("Solution is not correct")
        elif self.test_prob == "two-clique":
            assert np.all(inliers.astype(float) == self.x.flatten()), ValueError("Solution is not correct")
        elif self.test_prob == "bunny":
            # Solve using original clipper
            self.clipper.solve_as_msrc_sdr()
            soln = self.clipper.get_solution()
            inliers_clipper = np.zeros(self.prob.size)
            inliers_clipper[soln.nodes] = 1
            assert np.all(inliers == inliers_clipper), ValueError("Python SDP solution does not match CLIPPER SDR solution")



if __name__ == "__main__":
    test = TestClipper(test_prob="bunny")
    # test.test_affine_constraints()
    # test.test_solve_fusion()
    # test.test_symb_fact(plot=False)
    test.test_solve_fusion()
    # test.test_solve_fusion_sparse()

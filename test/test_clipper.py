import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.special import gammainc
import matplotlib.pyplot as plt
import cvxpy as cp
import mosek.fusion as fu
import scipy.sparse as sp
import unittest

from src.clipper.clipper import ConsistencyGraphProb, generate_dataset, get_err

class TestClipper(unittest.TestCase):
    def __init__(self):
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
        self.D1, self.D2, self.Agt, self.A = generate_dataset(
            self.pcfile, self.m, self.n1, self.n2o, self.outrat, self.sigma, self.T_21
        )
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
        # Define problem
        self.prob = ConsistencyGraphProb(self.D1, self.D2, self.A)
        
        
    def test_affinity_matrix(self):
        affinity_matrix = self.prob.get_affinity_matrix()
        self.assertEqual(affinity_matrix.shape[0], self.n1)
        self.assertEqual(affinity_matrix.shape[1], self.n1)

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
        
        # Process solution
        inliers, er, x_opt = self.prob.process_sdp_var(X)
        # Check ER
        assert er > 1e6, ValueError("Solution not rank-1")
    
    def test_symb_fact(self):
        self.prob.symb_fact_affinity()
        self.assertIsNotNone(self.prob.symb)    
    
    def test_solve_fusion_sparse(self):
        X, info = self.prob.solve_fusion_sparse(verbose=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        
        # Process solution
        inliers, er, x_opt = self.prob.process_sdp_var(X)
        # Check ER
        assert er > 1e6, ValueError("Solution not rank-1")
         



if __name__ == "__main__":
    test = TestClipper()
    # test.test_affine_constraints()
    # test.test_solve_fusion()
    # test.test_symb_fact()
    # test.test_solve_fusion()
    test.test_solve_fusion_sparse()

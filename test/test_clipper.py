import sys
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
import unittest

from src.clipper.clipper import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points, mat2hvec_ind, PARAMS_SCS_DFLT, DDConeVar, hvec2mat, prune_affinity
from src.clipper.rank_reduction import get_low_rank_factor


class TestClipper(unittest.TestCase):
    def __init__(self, test_prob="three-clique", seed=0, outrat=0.9, threshold=0.0):
        self.test_prob = test_prob
        np.random.seed(seed)
        if test_prob == "bunny":
            self.init_bunny_test(outrat=outrat, threshold=threshold)
        elif test_prob == "two-clique":
            self.two_clique_test()
        elif test_prob == "three-clique":
            self.three_clique_test()
        else:
            raise ValueError("Problem not recognized.")

        # Define problem
        self.prob = ConsistencyGraphProb(self.affinity)
        self.size = self.affinity.shape[0]

    def two_clique_test(self):
        """Simple test with two separate cliques"""
        self.affinity = sp.linalg.block_diag(np.ones((5, 5)), np.ones((3, 3)))
        # self.affinity =
        self.x = np.vstack([np.ones((5, 1)), np.zeros((3, 1))])
        self.X = self.x @ self.x.T

    def three_clique_test(self):
        """Simple test with three overlapping cliques"""
        self.affinity = sp.linalg.block_diag(np.ones((5, 5)), np.ones((3, 3)))
        self.affinity[4, 5] = 1
        self.affinity[5, 4] = 1

        # self.affinity =
        self.x = np.vstack([np.ones((5, 1)), np.zeros((3, 1))])
        self.X = self.x @ self.x.T

    def init_bunny_test(self, outrat, threshold=0.0):
        # Set up common variables for tests
        self.m = 100
        self.n1 = 100
        self.n2o = 10
        self.outrat = outrat
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
        self.affinity, self.clipper = get_affinity_from_points(
            self.D1, self.D2, self.A, threshold=threshold)
        # Generate a solution vector
        x = np.zeros((self.A.shape[0], 1))
        for i, a in enumerate(self.A):
            # Find any associations that are in the GT set
            if np.any(np.sum(a == self.Agt, axis=1) == 2):
                x[i] = 1
            else:
                x[i] = 0
        self.x = x
        # Generate a solution matrix
        self.X = x @ x.T

    def test_affine_constraints(self, X=None, tol=1e-10):
        eqs, ineqs = self.prob.get_affine_constraints()
        self.assertIsInstance(eqs, list)
        self.assertIsInstance(ineqs, list)
        self.assertGreater(len(eqs) + len(ineqs), 0)
        if X is None:
            X = self.X
        # Test Equalities
        for i, (A, b) in enumerate(eqs):
            assert np.abs(np.trace(
                A @ self.X) + b) < tol, ValueError(f"Equality Constraint {i} violated!")
        # Test Inequalities
        for i, (A, b) in enumerate(eqs):
            assert np.trace(A @ self.X) + b > - \
                tol, ValueError(f"Equality Constraint {i} violated!")

    def test_solve_cvxpy(self):
        X, H, yvals = self.prob.solve_cvxpy(verbose=False)
        self.assertIsNotNone(X)
        self.assertIsNotNone(H)
        self.assertIsInstance(yvals, list)

    def test_solve_fusion(self):
        # Check that solution is correct
        X, info = self.prob.solve_fusion(verbose=True, ineq=False, homog=False)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        self.check_solution(X, homog=False)
        # Get optimal solution and cost
        cost = info['cost']
        inliers, er, x_opt = self.prob.process_sdp_var(X)
        x_opt_h = np.hstack([x_opt, 1.0])

        # Check solution when homogenizing
        X_h, info_h = self.prob.solve_fusion(verbose=True, homog=True)
        self.assertIsNotNone(X_h)
        self.assertIn("success", info_h)
        self.assertTrue(info_h["success"])
        # Check that costs are the same
        self.assertTrue(info['cost'] - info_h["cost"] < 1e-6)
        # Check that optimal solution is in the null space of the certificate
        np.testing.assert_allclose(
            info_h["H"]@x_opt_h, np.zeros(x_opt_h.shape), atol=1e-4)
        # Process to get low rank factor
        V = self.prob.reduce_rank(X_h, info_h)
        X_h_r = V @ V.T
        self.check_solution(X_h_r, homog=True)

    def test_solve_fusion_homog_cost(self):
        """Puts the cost """
        X, info = self.prob.solve_fusion(
            verbose=True, homog=True, homog_cost=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        self.check_solution(X, homog=True)

    def test_solve_fusion_dense(self):
        X, info = self.prob.solve_fusion(verbose=True, dense_cost=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        self.check_solution(X)

        # Check homogenized version
        X_h, info_h = self.prob.solve_fusion(
            verbose=True,
            homog_cost=False,
            dense_cost=True,
            homog=True)
        self.assertIsNotNone(X_h)
        self.assertIn("success", info_h)
        self.assertTrue(info_h["success"])
        # self.check_solution(X)

    def test_scs_setup(self):
        cone, data = self.prob.get_scs_setup()
        ncvars = cone['l'] + np.sum([n*(n+1)/2 for n in cone['s']])
        nclqvars = 0
        for c in self.prob.cliques:
            nclqvars += len(c)*(len(c)+1)/2
        assert data['A'].shape == (data['b'].shape[0], data['c'].shape[0]), ValueError(
            "A matrix has wrong shape")
        if self.test_prob == "two-cliques":
            npos = 5*6/2 + 3*4/2+1
            assert data['A'].shape[1] == npos, ValueError(
                "wrong number of constraints")
            assert data['A'].shape[0] == npos + \
                nclqvars, ValueError("wrong number of variables")
        if self.test_prob == "three-clique":
            npos = 5*6/2 + 1 + 3*4/2 + 1
            assert data['A'].shape[1] == npos, ValueError(
                "wrong number of constraints")
            assert data['A'].shape[0] == npos + \
                nclqvars, ValueError("wrong number of variables")

    def test_mat2vec_ind(self, n=10):
        cols, rows = np.triu_indices(n)
        inds = list(zip(rows, cols))
        vec_inds = [mat2hvec_ind(n, row, col) for row, col in inds]
        assert np.all(vec_inds == list(range(len(vec_inds)))), ValueError(
            "Vectorization indexing not working")

    def test_solve_scs(self):
        X, info = self.prob.solve_scs_sparse(verbose=True)
        self.check_solution(X)

    def test_solve_scs_homog(self):
        # Set up params
        scs_params = PARAMS_SCS_DFLT
        scs_params['verbose'] = True
        X, info = self.prob.solve_scs_sparse(
            setup_kwargs=scs_params, homog=True)
        # Retrieve dual solution
        H_c_list, H = self.prob.get_dual_sol(info, homog=True)

        # Add constraint info
        info['constraints'] = self.prob.get_affine_constraints_homog()
        # # Process to get low rank factor
        V = self.prob.reduce_rank(X, info)
        X_h_r = V @ V.T
        self.check_solution(X_h_r, homog=True)

    def test_symb_fact(self, plot=False):
        self.prob.symb_fact_affinity()
        self.assertIsNotNone(self.prob.symb)
        # test clique definitions
        if self.test_prob == "two-clique":
            # Only two cliques
            assert len(self.prob.cliques) == 2, ValueError(
                "This test should only have two cliques")
            assert self.prob.cliques == [[0, 1, 2, 3, 4], [5, 6, 7]]
            mapping = {0: {0}, 1: {0}, 2: {0}, 3: {
                0}, 4: {0}, 5: {1}, 6: {1}, 7: {1}}
            assert self.prob.clq_lookup == mapping, ValueError(
                "index to clique mapping incorrect")
        elif self.test_prob == "three-clique":
            assert len(self.prob.cliques) == 3, ValueError(
                "This test should only have two cliques")
            cliques = [set([0, 1, 2, 3, 4]), set([4, 5]), set([5, 6, 7])]
            for clique in self.prob.cliques:
                assert set(clique) in cliques, ValueError("Cliques Incorrect")

        # Test ind to clique map
        for key, clq_list in self.prob.clq_lookup.items():
            for clq_ind in clq_list:
                assert key in self.prob.cliques[clq_ind], ValueError(
                    "index to clique map is broken")
        # Check that fill edges are not in original sparsity pattern
        for edge_fill in self.prob.fill_edges:
            assert edge_fill not in self.prob.edges, ValueError(
                "Fill edge is in original edge set")

        if plot:
            G = nx.Graph()  # Use nx.DiGraph() for a directed graph
            G.add_edges_from(self.prob.edges)
            plt.figure(figsize=(5, 5))
            pos = nx.kamada_kawai_layout(G)
            nx.draw(G, pos, with_labels=False, node_color='blue',
                    edge_color='gray', node_size=5)
            plt.show()

    def test_solve_fusion_sparse(self):
        X, info = self.prob.solve_fusion_sparse(verbose=True)
        self.assertIsNotNone(X)
        self.assertIn("success", info)
        self.assertTrue(info["success"])
        self.check_solution(X)

    def test_fusion_dual_homog(self, tol=1e-5):

        # SDP Cone
        H_sdp, info_sdp = self.prob.solve_fusion_dual_homog(verbose=True)
        # DD Cone
        H_dd, info_dd = self.prob.solve_fusion_dual_homog(
            verbose=True, cone="DD")
        # SDD Cone
        H_sdd, info_sdd = self.prob.solve_fusion_dual_homog(
            verbose=True, cone="SDD")

        # Check that dual is corank 2
        corank = np.sum(np.linalg.eigvalsh(H_sdp) < tol)
        assert corank == 2, ValueError("SDP solution should have corank 2")

        # Check dual costs
        assert np.abs(info_sdp['cost'] - np.sum(self.x)) < tol
        assert info_sdd['cost'] - info_sdp['cost'] >= - \
            tol, ValueError(
                "SDD approximation should have higher cost than SDP")
        assert info_dd['cost'] - info_sdd['cost'] >= - \
            tol, ValueError(
                "DD approximation should have higher cost than SDD")

    def test_dd_change_basis(self, tol=1e-5):
        """test diagonal dominant with change of basis"""

        # SDP Cone
        # H_sdp, info_sdp = self.prob.solve_fusion_dual_homog(verbose=True)
        # DD Cone
        H_dd, info_dd = self.prob.solve_fusion_dual_homog(
            verbose=True, cone="DD")

        for i in range(5):
            U = np.linalg.cholesky(H_dd+np.eye(H_dd.shape[0])*1e-6)
            U = sp.sparse.csc_array(U)
            U.eliminate_zeros()
            H_dd, info_dd = self.prob.solve_fusion_dual_homog(
                verbose=True, cone="DD", U=U.T)

        print("done")

    def test_dd_basis(self):
        """Test the DD basis vector generation"""
        dim = 50
        V = DDConeVar.get_basis_vectors(dim=dim)
        assert V.shape == (dim*(dim+1)/2, dim **
                           2), ValueError("Basis shape not correct")
        # Check diagonal dominance
        hvec = (V @ np.random.rand(dim**2)[:, None]).flatten()
        mat = hvec2mat(hvec)
        for i in range(dim):
            thesum = np.sum(np.abs(mat[:, i])) - np.abs(mat[i, i])
            assert mat[i, i] > thesum, ValueError(
                "Matrix not diagonally dominant")

        # Check version with change of basis
        U = np.random.randn(dim, dim) * 5
        V = DDConeVar.get_basis_vectors(dim=dim, U=U)
        assert V.shape == (dim*(dim+1)/2, dim **
                           2), ValueError("Basis shape not correct")
        # Check diagonal dominance
        hvec = (V @ np.random.rand(dim**2)[:, None]).flatten()
        Uinv = np.linalg.inv(U)
        mat = Uinv.T @ hvec2mat(hvec) @ Uinv

        for i in range(dim):
            thesum = np.sum(np.abs(mat[:, i])) - np.abs(mat[i, i])
            assert mat[i, i] > thesum, ValueError(
                "Matrix not diagonally dominant")

    def test_pruning(self):
        # Generate graph
        n = 9
        edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [
                         3, 4], [4, 5], [4, 6], [4, 7], [4, 8], [1, 8]])
        affinity = sp.sparse.csc_array(
            (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
        affinity = affinity + affinity.T
        affinity = affinity + sp.sparse.eye(n)
        # Get pruned affinity matrix
        new_affinity = prune_affinity(affinity=affinity, clique_size_lb=3)
        assert new_affinity.shape == (6, 6), ValueError(
            "Returned wrong affinity matrix")
        new_affinity = prune_affinity(affinity=affinity, clique_size_lb=4)
        assert new_affinity.shape == (4, 4), ValueError(
            "Returned wrong affinity matrix")
        assert np.all(new_affinity.toarray() == 1), ValueError(
            "Returned wrong affinity matrix")

        # NOTE: We assume the matrix is thresholded
        # Solve the problem to get max clique approx
        self.clipper.solve()
        soln = self.clipper.get_solution()
        max_clq_sz = len(soln.nodes)
        # Prune
        affinity = self.prob.affinity
        affinity_new = prune_affinity(affinity, max_clq_sz)
        # Get degree+1
        degree = np.sum(affinity_new, axis=0)
        # Check shapes
        assert affinity_new.shape[0] > 0, ValueError(
            "Pruned affinity is empty")
        assert np.all(degree - \
            max_clq_sz >= 0), ValueError(
                "Minimum degree should be greater than (max clique size)-1")

    def test_ddstar(self, tol=1e-3):
        scs_params = PARAMS_SCS_DFLT
        scs_params['verbose'] = False

        # # Run for one iteration
        # X, info_p = self.prob.solve_ddstar_cut(scs_params={}, max_iter=1)
        # # Compare to dual solution
        # H, info_d = self.prob.solve_fusion_dual_homog(
        #     verbose=True, cone="DD")
        # assert np.abs(-info_p['info']['pobj'] - info_d['cost']) < tol, ValueError(
        #     "Primal and Dual versions of DD problem have different cost")

        # Run for multiple iterations
        X, info_p = self.prob.solve_ddstar_cut(scs_params=scs_params)
        # Compare to dual solution of SDP
        H, info_d = self.prob.solve_fusion_dual_homog(
            verbose=True, cone="SDP")
        assert np.abs(-info_p['info']['pobj'] - info_d['cost']) < tol, ValueError(
            "Primal and Dual versions of DD problem have different cost")

    def check_solution(self, X, homog=False):
        """Check that solution is rank 1 and that the solution matches what we expect."""
        # Process solution
        inliers, er, x_opt = self.prob.process_sdp_var(X, homog=homog)
        # Check ER
        assert er > 1e6, ValueError("Solution not rank-1")

        if self.test_prob == "two-clique":
            assert np.all(inliers.astype(float) == self.x.flatten()
                          ), ValueError("Solution is not correct")
        elif self.test_prob == "three-clique":
            assert np.all(inliers.astype(float) == self.x.flatten()
                          ), ValueError("Solution is not correct")
        elif self.test_prob == "bunny":
            # Solve using original clipper
            self.clipper.solve_as_msrc_sdr()
            soln = self.clipper.get_solution()
            inliers_clipper = np.zeros(self.prob.size)
            inliers_clipper[soln.nodes] = 1
            assert np.all(inliers == self.x.T), ValueError(
                "Python SDP solution does not match CLIPPER SDR solution")


if __name__ == "__main__":
    test = TestClipper(test_prob="bunny", outrat=0.9, threshold=0.8)
    # test.test_affine_constraints()
    # test.test_solve_fusion()
    # test.test_solve_fusion_dense()
    # test.test_solve_fusion_homog_cost()
    # test.test_symb_fact(plot=False)
    # test.test_solve_fusion()
    # test.test_solve_fusion_sparse()
    # test.test_scs_setup()
    # test.test_mat2vec_ind()
    # test.test_solve_scs()
    # test.test_solve_scs_homog()
    # test.test_fusion_dual_homog()
    # test.test_dd_change_basis()
    # test.test_dd_basis()
    # test.test_ddstar()
    test.test_pruning()

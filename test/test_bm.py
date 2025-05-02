import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import unittest

from src.clipper.clipper import generate_bunny_dataset, get_affinity_from_points, prune_affinity
from src.clipper.burer_monteiro import CGraphProb, solve_bm_ipopt
from src.clipper.burer_monteiro_ineq import solve_bm_ipopt_ineq


class TestBM(unittest.TestCase):
    def __init__(self, test_prob="three-clique", seed=0, outrat=0.9, threshold=0.0):
        self.test_prob = test_prob
        np.random.seed(seed)
        if test_prob == "bunny":
            self.bunny_setup(outrat=outrat, threshold=threshold)
        elif test_prob == "two-clique":
            self.two_clique_setup()
        elif test_prob == "three-clique":
            self.three_clique_setup()
        else:
            raise ValueError("Problem not recognized.")

        # Define problem
        self.dim = self.affinity.shape[0]
        self.rank = 2
        self.prob = CGraphProb(self.affinity, rank=self.rank)
        # True solution
        n_inlier = np.sum(self.x)
        self.sol = self.x.flatten() / np.sqrt(n_inlier)
        self.vecV = np.hstack([self.sol, np.zeros(self.dim*(self.rank-1))])

    def two_clique_setup(self):
        """Simple test with two separate cliques"""
        block1 = 2
        block2 = 2
        self.affinity = sp.linalg.block_diag(
            np.ones((block1, block1)), np.ones((block2, block2)))
        # self.affinity =
        self.x = np.hstack([np.ones(block1), np.zeros(block2)])

    def three_clique_setup(self):
        """Simple test with three overlapping cliques"""
        self.affinity = sp.linalg.block_diag(np.ones((5, 5)), np.ones((3, 3)))
        self.affinity[4, 5] = 1
        self.affinity[5, 4] = 1

        # self.affinity =
        self.x = np.vstack([np.ones((5, 1)), np.zeros((3, 1))])
        self.X = self.x @ self.x.T

    def bunny_setup(self, outrat, mult=1, threshold=0.0):
        # Set up common variables for tests
        self.m = 100*mult
        self.n1 = 100*mult
        self.n2o = 10*mult
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
        x = np.zeros(self.A.shape[0])
        for i, a in enumerate(self.A):
            # Find any associations that are in the GT set
            if np.any(np.sum(a == self.Agt, axis=1) == 2):
                x[i] = 1
            else:
                x[i] = 0
        self.x = x

    def test_objective(self):

        obj_val = self.prob.objective(self.vecV)
        # Test
        obj_val_true = self.sol @ self.affinity @ self.sol
        assert obj_val == obj_val_true, ValueError(
            "objective value is incorrect")

    def test_grad(self):
        # Obj is tr( V.T M V), grad is 2 V.T @ M
        grad = self.prob.gradient(self.vecV)
        # Test
        obj_val = grad @ self.vecV
        obj_val_true = self.sol @ self.affinity @ self.sol
        assert obj_val == 2 * obj_val_true, ValueError(
            "grad @ V should be equal to 2x objective at solution")

    def test_constraints(self, tol=1e-5):

        # Number of constraints
        n_cons = len(self.prob.nonedges) + 1
        vals_cons = self.prob.constraints(self.vecV)
        assert len(vals_cons) == n_cons, ValueError(
            "Wrong number of constraints")
        vals_gt = np.zeros(n_cons)
        vals_gt[-1] = 1.0
        np.testing.assert_allclose(
            vals_cons, vals_gt, err_msg="Constraint violation incorrect")

        # Violate trace
        vals_cons = self.prob.constraints(self.vecV*2)
        assert vals_cons[-1] > 1 + \
            tol, ValueError("trace constraint should be violated")

        # Violate clique
        inds = np.where(self.vecV == 0)
        for ind in inds:
            vecV_viol = self.vecV.copy()
            vecV_viol[ind] = self.vecV[0]
            vals_cons = self.prob.constraints(vecV_viol)
            assert np.any(
                np.array(vals_cons)[:-1] > tol), ValueError("clique constraint should be violated")

    def test_jacobian(self):
        """Test the Jacobian generation"""
        # Number of constraints
        n_cons = len(self.prob.nonedges) + 1
        # Check dimensions on jacobian
        rows, cols = self.prob.jacobianstructure()
        vals = self.prob.jacobian(self.vecV)
        assert len(rows) == len(cols), ValueError(
            "rows and cols not same length")
        assert len(rows) == len(vals), ValueError(
            "vals and rows not same length")
        # Build jacobian
        J = sp.sparse.csc_array((vals, (rows, cols)),
                                shape=(n_cons, self.dim*self.rank))
        # Get empirical Jacobian
        eps = 1e-6
        diff = []
        cons_vals = np.array(self.prob.constraints(self.vecV))
        for i in range(len(self.vecV)):
            vecV_pert = self.vecV.copy()
            vecV_pert[i] += eps
            cons_vals_pert = np.array(self.prob.constraints(vecV_pert))
            diff.append((cons_vals_pert - cons_vals)/eps)
        J_emp = np.vstack(diff).T
        # Compare analytic and empirical Jacobian
        np.testing.assert_allclose(J.toarray(
        ), J_emp, atol=2*eps, err_msg="constraint jacobian not matching empirical jacobian")

    def test_hessian(self):
        """Test Hessian generation"""
        # Number of constraints
        n_cons = len(self.prob.nonedges) + 1
        mults = range(1, n_cons+1)
        obj_factor = 6
        # Generate Hessian matrix manually
        H_k = -self.affinity * obj_factor
        # Sparsity constraints
        ind = 0
        for i, j in self.prob.nonedges:
            H_k[i, j] = mults[ind]
            H_k[j, i] = mults[ind]
            ind += 1
        # Add final trace constraint multiplier
        H_k += np.eye(self.dim)*mults[ind]
        H_true = np.kron(np.eye(self.rank), H_k)
        # Generate Hessian
        h_dim = self.dim*self.rank
        rows, cols = self.prob.hessianstructure()
        vals = self.prob.hessian(
            self.vecV, lagrange=mults, obj_factor=obj_factor)
        H = sp.sparse.csc_array((vals, (rows, cols)), shape=(h_dim, h_dim))
        # Convert to dense and fill upper triangle
        H = H.toarray()
        H += H.T
        H[range(h_dim), range(h_dim)] /= 2
        # Check values
        np.testing.assert_allclose(
            H, H_true, err_msg="Hessian not matching manual generation")

    def test_ipopt(self):
        """Test IPOPT solver"""
        if self.test_prob == "bunny":
            # Get initialization from clipper
            self.clipper.solve()
            soln = self.clipper.get_solution()
            max_clq_sz = len(soln.nodes)
            # Prune the graph
            affinity, keep_inds = prune_affinity(self.affinity, max_clq_sz)
            # Get new solution
            x = np.zeros(affinity.shape[0])
            inds = [np.where(keep_inds == node)[0][0] for node in soln.nodes]
            x[inds] = 1 / np.sqrt(max_clq_sz)
        else:
            x = self.x/np.sqrt(np.sum(self.x))
            affinity = self.affinity

        # Solve BM
        rank = 3
        x0 = np.hstack([x/rank]*rank)
        # x0 = np.random.rand(len(x)*rank)
        # Run bm method
        V, H, info = solve_bm_ipopt(
            affinity, rank=rank, x0=x0, dense_cost=True)

        # Check that Hessian is PSD in the feasible directions.
        # Find active constraints
        active = np.where(np.abs(info["mult_g"]) > 1e-2)[0]
        # Get Jacobian at solution
        dim = V.shape[0]
        x = V.T.flatten()
        J = info['prob'].build_jac(x)
        # Get null space of jacobian
        U, S, _ = np.linalg.svd(J[active, :dim].T.toarray())
        inds = np.where(S < 1e-3)[0]
        Z = U[:, inds]
        evals = np.linalg.eigvalsh(Z.T @ H @ Z)
        assert np.all(
            evals > -1e-6), ValueError("Hessian should be PSD in feasible directions")
        print("done")

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
    # SETUP
    test = TestBM(test_prob="bunny", outrat=0.2, threshold=0.5)
    # test = TestBM(test_prob="three-clique")
    # test = TestBM(test_prob="two-clique")

    # TESTS
    # test.test_objective()
    # test.test_grad()
    # test.test_constraints()
    # test.test_jacobian()
    # test.test_hessian()
    test.test_ipopt()

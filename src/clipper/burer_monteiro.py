import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from time import time
import cyipopt

from src.clipper.utils import *


class CGraphProb():
    def __init__(self, affinity, rank, dense_cost=False, verbose=False):
        self.affinity = affinity
        self.dim = affinity.shape[0]
        self.rank = rank
        self.dense_cost = dense_cost  # Dense cost flag (CLIPPER+ Setup)

        # Process the edges of the affinity matrix
        self.edges = []
        self.nonedges = []
        for j in range(self.dim):
            for i in range(j, self.dim):
                if self.affinity[i, j] == 0:
                    self.nonedges.append([i, j])
                    if self.dense_cost:
                        # Fill in empty values with negatives
                        self.affinity[i, j] = -self.dim
                        self.affinity[j, i] = -self.dim
                else:
                    self.edges.append([i, j])
        # Num constraints
        self.ncons = len(self.nonedges) + 1
        # Verbosity
        self.verbose = verbose

    def reshape_x(self, x):
        return np.reshape(x, (self.rank, self.dim)).T

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        # Reshape x into matrix form
        V = self.reshape_x(x)
        objsum = []
        for i in range(self.rank):
            objsum.append(-V[:, [i]].T @ self.affinity @ V[:, [i]])

        return np.sum(objsum)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        I_r = sp.eye(self.rank)
        return -2*sp.kron(I_r, self.affinity) @ x

    def constraints(self, x):
        """Returns the constraints."""
        # Get matrix
        V = self.reshape_x(x)
        # Sparsity constraints
        constraint_vals = []
        for i, j in self.nonedges:
            constraint_vals.append(
                np.sum([2*V[i, k]*V[j, k] for k in range(self.rank)]))
        # Trace Constraint
        constraint_vals.append(
            np.sum([V[:, [k]].T*V[:, k] for k in range(self.rank)]))

        return constraint_vals

    def jacobianstructure(self):
        """Returns the row and column indices of thenon-zero valuse of the Jacobian"""
        # Init
        rows, cols = [], []
        curr_row = 0
        # Sparsity constraints
        # NOTE: J = vec(V).T @ (I_r kron A_ij) where A_ij is all zeros except at (i,j), (j,i)
        for i, j in self.nonedges:
            for k in range(self.rank):
                cols.append(k*self.dim + i)
                rows.append(curr_row)
                cols.append(k*self.dim + j)
                rows.append(curr_row)
            curr_row += 1
        # Trace Constraints
        # NOTE J = vec(V).T
        for i in range(self.dim*self.rank):
            cols.append(i)
            rows.append(curr_row)

        return rows, cols

    def jacobian(self, x):
        """Returns the Jacobian of the constraints with respect to x."""
        # Get matrix variable
        V = self.reshape_x(x)

        # Sparsity constraints
        vals = []
        for i, j in self.nonedges:
            for k in range(self.rank):
                # NOTE: ordering is important here, needs to match the structure
                vals.append(2 * V[j, k])
                vals.append(2 * V[i, k])
        # Trace gradient is just x
        vals = np.hstack([vals, 2*x])

        return vals

    def hessianstructure(self):
        """Returns the row and column indices for non-zero vales of the
        Hessian."""
        # NOTE: Hessian = (I_r kron H_k)
        # H_k = -M + sum(A_ij * lambda_ij) + I * lambda_t
        # Each H_k will be a dense matrix.

        # Build dense H_k structure
        rows_1, cols_1 = zip(*self.edges)
        if len(self.nonedges):
            rows_2, cols_2 = zip(*self.nonedges)
            rows = rows_1 + rows_2
            cols = cols_1 + cols_2
        else:
            rows = rows_1
            cols = cols_1

        # Repeat structure along diagonal
        rows_all, cols_all = [], []
        for k in range(self.rank):
            rows_all += [val + self.dim*k for val in rows]
            cols_all += [val + self.dim*k for val in cols]

        return rows_all, cols_all

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        ind_tr_cons = self.ncons-1
        # Build Dense H_k matrix
        vals = []
        # Edges
        for i, j in self.edges:
            # Add affinity matrix term
            val = -self.affinity[i, i]*obj_factor
            if i == j:
                # Diagonal entry, add trace constraint multiplier
                val += lagrange[ind_tr_cons]
            vals.append(val)

        # Non Edges
        for k, (i, j) in enumerate(self.nonedges):
            # Off diagonal, affinity zero, add multiplier
            val = lagrange[k]
            # If using dense cost, fixed term (see CLIPPER+)
            if self.dense_cost:
                val += self.dim * obj_factor
            vals.append(val)

        # Repeat H_k "rank" times
        vals = vals * self.rank

        return vals

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):
        """Prints information at every Ipopt iteration."""
        if self.verbose:
            print(f"Iter {iter_count}, Obj: {obj_value}")
            msg = "Objective value at iteration #{:d} is - {:g}"
            print(msg.format(iter_count, obj_value))

    def recover_certificate(self, mults):
        # recover last hessian
        h_dim = self.dim*self.rank
        rows, cols = self.hessianstructure()
        vals = self.hessian(
            None, lagrange=mults, obj_factor=1)
        H = sp.csc_array((vals, (rows, cols)), shape=(h_dim, h_dim))
        # Just get one of the copies
        H = H[:, :self.dim][:self.dim, :]
        # Convert to full dense matrix
        H = H.toarray()
        H += H.T
        H[range(self.dim), range(self.dim)] /= 2

        return H

    def build_jac(self, x):
        """Construct the Jacobian"""
        rows, cols = self.jacobianstructure()
        vals = self.jacobian(x)
        # Build jacobian
        J = sp.csc_array((vals, (rows, cols)),
                                shape=(self.ncons, self.dim*self.rank))
        return J


def solve_bm_ipopt(affinity, rank=2, x0=None, dense_cost=False):
    """Solve the burer-monteiro using ipopt"""

    prob = CGraphProb(affinity=affinity, rank=rank, dense_cost=dense_cost)

    # Define variable bounds (set to values that are never attained)
    high_value = 1e20
    lb = None
    ub = None
    # Define constraint bounds
    cl = np.zeros(prob.ncons)
    cl[-1] = 1
    if dense_cost:
        # Change sparsity constraint equalities into inequalities
        cu = np.ones(prob.ncons)*high_value
        cu[-1] = 1
    else:
        cu = cl
    # init solution
    if x0 is None:
        x0 = np.zeros(prob.rank*prob.dim)

    # Define nonlinear program
    nlp = cyipopt.Problem(
        n=len(x0),
        m=len(cl),
        problem_obj=prob,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    # Options
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-5)
    # Turn on hessian approx
    nlp.add_option('hessian_approximation', 'limited-memory')
    # Solve
    x, info = nlp.solve(x0)
    # Recover matrix solution
    V = prob.reshape_x(x)
    # Recover certificate
    H = prob.recover_certificate(mults=info['mult_g'])
    # Add problem object to info
    info['prob'] = prob

    return V, H, info

import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.special import gammainc
import matplotlib.pyplot as plt
import cvxpy as cp
import mosek.fusion.pythonic as fu
import scipy.sparse as sp
import clipperpy
from cvxopt import amd, spmatrix, matrix
import chompack
from time import time
import scs
import networkx as nx

from src.clipper.rank_reduction import rank_reduction

PARAMS_SCS_DFLT = dict(max_iters=2000,
                       acceleration_interval=10,
                       acceleration_lookback=10,
                       eps_abs=1e-3,
                       eps_rel=1e-3,
                       eps_infeas=1e-7,
                       time_limit_secs=0,
                       verbose=False)


class DDConeVar():
    """Implements diagonally dominant cone variable for mosek fusion"""

    def __init__(self, mdl, dim, U=None, name="dd"):
        assert isinstance(mdl, fu.Model), ValueError(
            "Must provide fusion model to define DDConVar")
        self.dim = dim  # Dimension of matrix
        self.shape = (dim, dim)
        self.mdl = mdl  # Model
        # Change of basis matrix
        self.U = U
        # Define basis variables
        self.alpha = mdl.variable(
            "alpha", self.dim**2, fu.Domain.greaterThan(0.0))
        self.basis = self.get_basis_vectors(self.dim, U)
        # Define actual matrix
        basis_fu = mat_fusion(self.basis)
        self.hvec = basis_fu @ self.alpha

    @staticmethod
    def get_basis_vectors(dim, U=None):
        """Generate the DDCone Basis Vectors (scaled half-vectorized format)
        Args:
            dim: the dimension of the DD matrix represented by the basis
            U:   change of basis matrix to be applied to the DD matrix
                    X = U.T Q U"""
        # Define basis vectors as columns of a sparse matrix
        rows, cols, vals = [], [], []
        currCol = 0
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    ind = mat2vec_ind(dim, i, i)
                    rows.append(ind)
                    cols.append(currCol)
                    vals.append(1)
                    currCol += 1  # increment column
                elif j > i:
                    ind_i = mat2vec_ind(dim, i, i)
                    ind_j = mat2vec_ind(dim, j, j)
                    ind_ij = mat2vec_ind(dim, i, j)
                    ind_ji = mat2vec_ind(dim, j, i)
                    # positive off diag
                    rows += [ind_i, ind_j, ind_ij, ind_ji]
                    vals += [1, 1, 1, 1]
                    cols += [currCol] * 4
                    currCol += 1  # increment column
                    # negative off diag
                    rows += [ind_i, ind_j, ind_ij, ind_ji]
                    vals += [1, 1, -1, -1]
                    cols += [currCol] * 4
                    currCol += 1  # increment column
        # Build matrix
        shape = (dim**2, dim**2)
        basis = sp.csc_matrix((vals, (rows, cols)), shape=shape)
        # Multiply by change of basis matrix if available
        if U is not None:
            if sp.issparse(U):
                UkronU = sp.kron(U.T, U.T)
            else:
                UkronU = np.kron(U.T, U.T)
            basis = UkronU @ basis
        # Eliminate and rescale into half vectorization formulation
        E = vec2hvec(dim)
        basis = E @ basis

        return basis

    def level(self):
        """Return level value in matrix form"""
        return hvec2mat(self.basis @ self.alpha.level())

    def index(self, i, j):
        """Return the Mosek variable at the given index. Assume lower triangular"""
        row, col = i, j
        if i == j:
            scale = 1
        elif i < j:
            row, col = j, i
            scale = 1/np.sqrt(2)
        else:
            scale = 1/np.sqrt(2)
        ind = mat2hvec_ind(self.dim, row, col)
        return scale * self.hvec.index(ind)


class SDDConeVar():
    """Implements scaled diagonally dominant cone variable for mosek fusion"""

    def __init__(self, mdl, dim, name="sdd"):
        assert isinstance(mdl, fu.Model), ValueError(
            "Must provide fusion model to define DDConVar")
        self.dim = dim  # Dimension of matrix
        self.shape = (dim, dim)
        self.mdl = mdl  # Model
        vecsize = int(dim * (dim+1)/2)
        # Variable representing matrix
        self.vecvar = mdl.variable(name, vecsize)
        # Principle Minor Matrices (dim choose 2)
        # NOTE: M = [a b ; b c] then cone order is (a, c, b)
        n_pmm = dim*(dim-1)
        self.mvars = mdl.variable(
            name+"_rqc", fu.Domain.inRotatedQCone(n_pmm, 3))
        # Define a unique mapping for the minors
        # NOTE: We assume the minors are indexed in according to the lower off diagonals
        self.mmap = {}
        ind = 0
        for j in range(self.dim):
            for i in range(j, self.dim):
                if i == j:
                    continue
                else:
                    self.mmap[(i, j)] = ind
                    ind += 1

        # Generate matrix variable parameterized on the minors
        self.vecvar = self.generate_var()

    def generate_var(self):
        """Generate the basis corresponding to the  """
        vecvar = []
        for col in range(self.dim):
            for row in range(col, self.dim):
                if row == col:  # On diagonal, need to sum minors
                    m_var_list = []
                    for i in range(row):  # Preceding minors
                        m_ind = self.mmap[(row, i)]
                        m_var_list.append(self.mvars.index(m_ind, 1))
                    for i in range(row+1, self.dim):  # Succeding minors
                        m_ind = self.mmap[(i, row)]
                        m_var_list.append(self.mvars.index(m_ind, 0))
                    m_var = fu.Expr.add(m_var_list)
                else:  # Off diagonal, select corresponding minor
                    m_ind = self.mmap[(row, col)]
                    m_var = self.mvars.index(m_ind, 2) / np.sqrt(2)
                # Add to variable
                vecvar.append(m_var)
        return vecvar

    def level(self):
        """Return level value in matrix form"""
        vecvar = []
        for col in range(self.dim):
            for row in range(col, self.dim):
                if row == col:  # On diagonal, need to sum minors
                    m_var_list = []
                    for i in range(row):  # Preceding minors
                        m_ind = self.mmap[(row, i)]
                        m_var_list.append(self.mvars.index(m_ind, 1).level())
                    for i in range(row+1, self.dim):  # Succeding minors
                        m_ind = self.mmap[(i, row)]
                        m_var_list.append(self.mvars.index(m_ind, 0).level())
                    m_var = np.sum(m_var_list)
                else:  # Off diagonal, select corresponding minor
                    m_ind = self.mmap[(row, col)]
                    m_var = self.mvars.index(m_ind, 2).level() / np.sqrt(2)
                # Add to variable
                vecvar.append(m_var)

        # Fill lower triangle
        out = np.zeros(self.shape)
        cols, rows = np.triu_indices(self.dim)
        out[rows, cols] = np.hstack(vecvar)
        out[range(self.dim), range(self.dim)] /= 2

        return out + out.T

    def index(self, i, j):
        """Return the Mosek variable at the given index. Assume lower triangular"""
        if i < j:
            row, col = j, i
        else:
            row, col = i, j
        ind = mat2hvec_ind(self.dim, row, col)

        return self.vecvar[ind]


class ConsistencyGraphProb():
    def __init__(self, affinity, factor=True):
        # Init affinity matrix
        if sp.issparse(affinity):
            self.affinity = affinity
        else:
            self.affinity = sp.csc_array(affinity)
        
        self.size = self.affinity.shape[0]
        # Run Symbolic Factorization
        if factor:
            self.symb_fact_affinity()

        # Mosek options
        TOL = 1e-4
        self.options_cvxpy = {}
        self.options_cvxpy["mosek_params"] = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": TOL,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": TOL,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": TOL,
            "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_PRIMAL",  # has no effect
        }
        self.options_fusion = {
            "intpntMaxIterations": 500,
            "intpntCoTolPfeas": TOL,
            "intpntCoTolDfeas": TOL,
            "intpntCoTolMuRed": TOL,
            "intpntSolveForm": "primal",  # has no effect
        }

    def threshold_affinity(self, thresh=0.5):
        """Threshold the affinity matrix"""
        cols_thrsh, rows_thrsh = [], []
        M = self.affinity
        rows, cols = M.nonzero()
        vals = M.data
        for i in range(len(cols)):
            if vals[i] >= thresh:
                cols_thrsh.append(cols[i])
                rows_thrsh.append(rows[i])

        M_thrsh = sp.csc_array(
            (np.ones(len(cols_thrsh)), (rows_thrsh, cols_thrsh)), shape=M.shape)
        return M_thrsh

    def get_affine_constraints(self):
        """Generate all affine constraints for the problem
        Equalities assumed to be of form: <A, X> + b = 0 
        Inequalities assumed to be of form: <A, X> + b >= 0"""
        M = self.affinity
        n = M.shape[0]
        eqs = []
        ineqs = []
        for i in range(n):
            for j in range(i, n):
                # Construct selection matrix
                if i == j:
                    vals = [1]
                    rows = [i]
                    cols = rows
                else:
                    vals = [0.5, 0.5]
                    rows = [i, j]
                    cols = [j, i]
                A = sp.csr_array((vals, (rows, cols)), shape=(n, n))
                # Each element of X is either zero or non-negative, depending on M
                if M[i, j] == 0:
                    eqs.append((A, 0.0))
                else:
                    ineqs.append((A, 0.0))
        # Add trace constraint
        A = sp.identity(n, format="csr")
        ineqs.append((-A, 1.0))

        return eqs, ineqs

    def get_affine_constraints_homog(self):
        """Get the constraints corresponding to the SCS formulation below
        NOTE: the ordering matters here; needs to align with lagrange multipliers"""

        # Generate positive constraint matrix
        constraints = []
        # Loop through variables.
        for iVar in range(self.size):
            rows = [iVar, self.size]
            cols = [self.size, iVar]
            vals = [1, 1]
            A = sp.csr_array((vals, (rows, cols)),
                             shape=(self.size+1, self.size+1))
            constraints.append(A)

        # Trace constraint
        rows = list(range(self.size))
        cols = list(range(self.size))
        vals = np.ones(self.size)
        A = sp.csr_array((vals, (rows, cols)),
                         shape=(self.size+1, self.size+1))
        constraints.append(A)

        # Homogenizing Constraint
        A = sp.csr_array(([1], ([self.size], [self.size])),
                         shape=(self.size+1, self.size+1))
        constraints.append(A)

        # Fill in constraints
        for i, j in self.fill_edges:
            vals = [0.5, 0.5]
            rows = [i, j]
            cols = [j, i]
            A = sp.csr_array((vals, (rows, cols)),
                             shape=(self.size+1, self.size+1))
            constraints.append(A)
        return constraints

    @staticmethod
    def merge_cosmo(cp, ck, np, nk):
        """clique merge function from COSMO paper:
        https://arxiv.org/pdf/1901.10887

        Uses the metric:
        Cp^3  + Ck^3 - (Cp U Ck)^3 > 0

        Args:
            cp (int): clique order of parent
            ck (int): clique order of child
            np (int): supernode order of parent
            nk (int): supernode order of child
        """
        # Metric: Cp^3  + Ck^3 - (Cp + Nk)^3
        return cp**3 + ck**3 > (cp + nk) ** 3

    def symb_fact_affinity(self, order=None, merge_func=None):
        """Generates the symbolic factorization of the affinity matrix.
        This factorization generates a clique tree for the associated graph. 
        Key members: 
            symb.cliques: list of cliques, each clique is a list of indices, 
            symb.seperators: list of seperator indices between a given clique and its parent
            symb.parent: list of the parents of each clique
        Cliques are listed in reverse topological order.

        """
        # Convert adjacency to sparsity pattern
        rows, cols = self.affinity.nonzero()
        # NOTE: only store edges on lower triangle to avoid double counting
        self.edges = list(zip(*sp.tril(self.affinity).nonzero()))
        self.pattern = spmatrix(1.0, rows, cols, self.affinity.shape)
        # get information from factorization
        if merge_func == "cosmo":
            merge_function = self.merge_cosmo
        else:
            merge_function = None
        if order is None:
            order = amd.order
        else:
            order = matrix(order)
        self.symb = chompack.symbolic(
            self.pattern, p=order, merge_function=merge_function)
        self.cliques = self.symb.cliques()
        self.sepsets = self.symb.separators()
        self.parents = self.symb.parent()
        # Get variable list in permuted order (symbolic factorization reorders things)
        var_list = list(range(self.size))
        var_list_perm = [var_list[p] for p in self.symb.p]
        self.clq_lookup = dict()
        self.clq_var_lookups = []
        for iClq, clique in enumerate(self.cliques):
            # Get the cliques and separator sets in the original ordering
            self.cliques[iClq] = [var_list_perm[v] for v in clique]
            self.sepsets[iClq] = set([var_list_perm[v]
                                     for v in self.sepsets[iClq]])
            # Generate mapping from index to clique mapping as a dictionary of lists
            for ind in self.cliques[iClq]:
                if ind not in self.clq_lookup.keys():
                    self.clq_lookup[ind] = set([iClq])
                else:
                    self.clq_lookup[ind].add(iClq)
            # For each clique, define a lookup table that maps global index to clique index.
            clq_var_lookup = {value: index for index,
                              value in enumerate(self.cliques[iClq])}
            self.clq_var_lookups.append(clq_var_lookup)
        # Get a list of "fill-in" edges
        pattern_filled = self.symb.sparsity_pattern(reordered=False)
        fillin = pattern_filled - self.pattern
        self.fill_edges = []
        self.fill_pattern = cvxmat2sparse(fillin)
        self.fill_pattern.eliminate_zeros()
        # NOTE: only store edges on lower triangle to avoid double counting
        self.fill_edges = [tuple(edge) for edge in zip(
            *self.fill_pattern.nonzero()) if edge[0] >= edge[1]]

        # Keep filed pattern
        self.filled_pattern = cvxmat2sparse(pattern_filled)
        self.filled_pattern.eliminate_zeros()

    def solve_fusion(self, options=None, dense_cost=False, ineq=True, homog=False, threshold=0.0, homog_cost=False, verbose=False):
        """Solve the MSRC problem with Mosek Fusion"""
        if options is None:
            options = self.options_fusion
        n = self.size
        if homog:
            size = n + 1
        else:
            size = n

        # Set up cost matrix
        M = self.affinity

        if dense_cost and threshold == 0:
            threshold = 0.5

        if threshold > 0:
            M = self.threshold_affinity(thresh=threshold)

        # Get sparse data for M
        rows, cols = M.nonzero()
        vals = M.data
        rows = list(rows)
        cols = list(cols)
        vals = list(vals)

        if dense_cost:
            # Construct dense cost matrix
            n = self.size
            M_d = -n*np.ones((n, n))
            M_d[rows, cols] = np.ones(len(rows))
            # Pad with zeros if homogenizing
            if homog and homog_cost:
                M = np.block([[M_d, 2*n*np.ones((n, 1))],
                              [2*n*np.ones((1, n)), np.zeros((1, 1))]])
            elif homog:
                M = np.block([[M_d, np.zeros((n, 1))],
                              [np.zeros((1, n)), np.zeros((1, 1))]])
            else:
                M = M_d
        elif homog and homog_cost:
            # Put the cost on the homogenizing variables
            rows = list(range(n)) + n * [n]
            cols = n * [n] + list(range(n))
            vals = [1/2] * 2 * n
            M = fu.Matrix.sparse(size, size, rows, cols, vals)
        else:
            M = fu.Matrix.sparse(size, size, rows, cols, vals)

        constraints = []
        clist = []
        ineq = []
        with fu.Model("primal") as mdl:
            if verbose:
                print("Constructing Problem")
            t0 = time()
            # Create SDP variable
            X = mdl.variable("X", fu.Domain.inPSDCone(size))
            # Add constriants
            for i in range(n):
                for j in range(i, n):
                    # Construct selection matrix
                    if i == j:
                        vals = [1]
                        rows = [i]
                        cols = rows
                    else:
                        vals = [0.5, 0.5]
                        rows = [i, j]
                        cols = [j, i]
                    A = fu.Matrix.sparse(size, size, rows, cols, vals)
                    # Each element of X is either zero or non-negative, depending on M
                    if self.affinity[i, j] == 0:
                        # Equalities not required if dense cost is used
                        if not dense_cost:
                            constr = mdl.constraint(
                                X.index(i, j)*2, fu.Domain.equalsTo(0))
                            clist.append(constr)
                            constraints.append(sp.csr_array(
                                (vals, (rows, cols)), shape=(size, size)))
                            ineq.append(False)
                        if not homog and ineq:
                            # Enforce inequalities on variables if not homogenizing
                            constr = mdl.constraint(f"ineq_{i}_{j}", X.index(
                                i, j)*2, fu.Domain.greaterThan(0))
                            clist.append(constr)
                            constraints.append(sp.csr_array(
                                (vals, (rows, cols)), shape=(size, size)))
                            ineq.append(True)

            if homog:
                if ineq:
                    # Implement inequalities as homogenizing constraints.
                    for i in range(n):
                        rows = [i, n]
                        cols = [n, i]
                        vals = [0.5, 0.5]
                        A = fu.Matrix.sparse(size, size, rows, cols, vals)
                        constr = mdl.constraint(
                            X.index(i, n)*2, fu.Domain.greaterThan(0))
                        clist.append(constr)
                        constraints.append(sp.csr_array(
                            (vals, (rows, cols)), shape=(size, size)))
                        ineq.append(True)

                # Add Homogenizing constraint
                A_h = fu.Matrix.sparse(size, size, [n], [n], [1])
                clist.append(constr)
                constr = mdl.constraint(X.index(n, n), fu.Domain.equalsTo(1))
                constraints.append(sp.csr_array(
                    ([1], ([n], [n])), shape=(size, size)))
                ineq.append(False)
            # Add Trace Constraint
            vals = np.ones(n)
            inds = list(range(n))
            A_tr = fu.Matrix.sparse(size, size, inds, inds, vals)
            constr = mdl.constraint(fu.Expr.dot(
                A_tr, X), fu.Domain.equalsTo(1))
            clist.append(constr)
            constraints.append(sp.csr_array(
                (vals, (inds, inds)), shape=(size, size)))
            ineq.append(False)
            # Add affinity matrix objective
            mdl.objective(fu.ObjectiveSense.Minimize, fu.Expr.dot(M, -X))

            if verbose:
                mdl.setLogHandler(sys.stdout)
                mdl.writeTask("problem.ptf")

            for key, val in options.items():
                mdl.setSolverParam(key, val)

            mdl.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            t1 = time()
            mdl.solve()
            t2 = time()
            if mdl.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                cost = mdl.primalObjValue()
                H = np.reshape(X.dual(), (size, size))
                X = np.reshape(X.level(), (size, size))
                mults = [c.dual() for c in clist]
                msg = f"success with status {mdl.getProblemStatus()}"
                success = True
            else:
                cost = None
                H = None
                X = None
                msg = f"solver failed with status {mdl.getProblemStatus()}"
                success = False
                mults = None
            info = {
                "success": success,
                "cost": cost,
                "msg": msg,
                "H": H,
                "time_setup": t1-t0,
                "time_solve": t2-t1,
                "constraints": constraints,
                "mults": mults,
                "ineq": ineq,
            }
            return X, info

    def solve_fusion_dual_homog(self, options=None, cone='SDP', U=None, verbose=False):
        """Solve the homogenized dual problem using different cone approximations"""
        # Get options
        if options is None:
            options = self.options_fusion
        n = self.size+1
        with fu.Model("Dual") as mdl:
            # Construct Model with specified cone
            if cone == "SDP":
                H = mdl.variable("X", fu.Domain.inPSDCone(n))
            elif cone == "DD":
                H = DDConeVar(mdl=mdl, U=U, dim=n)
            elif cone == "SDD":
                H = SDDConeVar(mdl=mdl, dim=n)
            else:
                raise ValueError("cone input not recognized")
            # Define homogenization and trace constraint multipliers.
            lambda_t = mdl.variable("l_t")
            lambda_h = mdl.variable("l_h")
            # Define inequality constraint multipliers
            mu = mdl.variable("mu", self.size, fu.Domain.greaterThan(0.0))
            lambda_ij = []
            # NOTE: we only loop over half of the matrix
            for i in range(self.size):
                # Add affinity matrix constraints
                for j in range(i, self.size):
                    if i == j:
                        # Diagonals with trace constraint multiplier
                        value = lambda_t - self.affinity[i, i]
                    elif j > i:
                        # Off diagonals have multiplier only if affinity is zero
                        if self.affinity[i, j] == 0:
                            lambda_ij.append(mdl.variable(f"lam_{i}_{j}"))
                            value = lambda_ij[-1]
                        else:
                            # Otherwise just use affinity value
                            value = -self.affinity[i, j]
                    mdl.constraint(f"elem_{i}_{j}", value-H.index(
                        i, j), fu.Domain.equalsTo(0.0))
                # Inequality constraint multipliers
                mdl.constraint(f"ineq_{i}", H.index(
                    i, self.size)+mu[i], fu.Domain.equalsTo(0))
            # Homogenizing Constraint
            mdl.constraint(f"homog", H.index(
                self.size, self.size)-lambda_h, fu.Domain.equalsTo(0))
            # Gather variables
            variables = [mu] + [lambda_h] + [lambda_t] + lambda_ij
            # Define objective
            mdl.objective(fu.ObjectiveSense.Minimize, lambda_t + lambda_h)

            if verbose:
                mdl.setLogHandler(sys.stdout)
                mdl.writeTask(f"dual_homog_{cone}.ptf")

            for key, val in options.items():
                mdl.setSolverParam(key, val)

            mdl.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            mdl.solve()
            if mdl.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                cost = mdl.primalObjValue()
                if cone == "SDP":
                    X = np.reshape(H.dual(), (n, n))
                    H = np.reshape(H.level(), (n, n))
                elif cone in ["DD", "SDD"]:
                    X = None
                    H = H.level()
                mults = [var.level() for var in variables]
                msg = f"success with status {mdl.getProblemStatus()}"
                success = True
            else:
                cost = None
                H = None
                X = None
                msg = f"solver failed with status {mdl.getProblemStatus()}"
                success = False
                mults = None
            info = {
                "success": success,
                "cost": cost,
                "msg": msg,
                "H": H,
                "mults": mults,
            }
            return H, info

    def solve_fusion_sparse(self, options=None, verbose=False):
        """Solve the MSRC problem with Mosek Fusion while exploiting sparsity."""
        if options is None:
            options = self.options_fusion

        with fu.Model("dual") as mdl:
            if verbose:
                print("Constructing Problem")
            t0 = time()
            obj_list = []
            trace_con_list = []
            X_cs = []
            # Generate cliques variables and trace constraint
            for iClq, clique_inds in enumerate(self.cliques):
                n = len(clique_inds)
                # Create SDP variable
                X_cs.append(mdl.variable(f"X_c{iClq}", fu.Domain.inPSDCone(n)))
                # Trace constraint
                A = mat_fusion(sp.identity(n, format='csr'))
                trace_con_list.append(fu.Expr.dot(A, X_cs[-1]))

            # Enforce inequalities and construct objective on the edges of the association graph
            obj_list = []
            # NOTE: we assume here that edges correspond to the lower triangle of the affinity matrix
            for edge in self.edges:
                # Build expression
                clq_var_list = []
                for iClq, ind0, ind1 in self.get_clique_inds(edge):
                    # Add to list of clique variables
                    clq_var_list.append(X_cs[iClq].index(ind0, ind1))
                # Add doubly non-negative constraint
                clq_sum = fu.Expr.add(clq_var_list)
                mdl.constraint(clq_sum, fu.Domain.greaterThan(0))
                # Add to objective
                if edge[0] == edge[1]:
                    # Along Diagonal
                    obj_list.append(fu.Expr.mul(self.affinity[edge], clq_sum))
                else:
                    # Off-Diagonal (multiply by 2)
                    obj_list.append(fu.Expr.mul(
                        2*self.affinity[edge], clq_sum))
            # Enforce equalities on the fill-in edges
            for edge in self.fill_edges:
                # Build expression
                clq_var_list = []
                for iClq, ind0, ind1 in self.get_clique_inds(edge):
                    # Add to list of clique variables
                    clq_var_list.append(X_cs[iClq].index(ind0, ind1))
                # Add doubly non-negative constraint
                clq_sum = fu.Expr.add(clq_var_list)
                mdl.constraint(clq_sum, fu.Domain.equalsTo(0))

            # Trace constraint
            mdl.constraint(fu.Expr.add(trace_con_list), fu.Domain.lessThan(1))

            # Add affinity matrix objective
            mdl.objective(fu.ObjectiveSense.Maximize, fu.Expr.add(obj_list))

            if verbose:
                mdl.setLogHandler(sys.stdout)
                mdl.writeTask("problem_sparse.ptf")

            for key, val in options.items():
                mdl.setSolverParam(key, val)
            mdl.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            t1 = time()
            mdl.solve()
            t2 = time()
            if mdl.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                # Reconstruct Full Variable
                X = np.zeros(self.affinity.shape)
                for iClq, clique_inds in enumerate(self.cliques):
                    n = len(clique_inds)
                    X[np.ix_(clique_inds, clique_inds)
                      ] += np.reshape(X_cs[iClq].level(), (n, n))
                cost = mdl.primalObjValue()
                msg = f"success with status {mdl.getProblemStatus()}"
                success = True
            else:
                cost = None
                X = None
                msg = f"solver failed with status {mdl.getProblemStatus()}"
                success = False
            info = {"success": success, "cost": cost, "msg": msg,
                    "time_setup": t1-t0, "time_solve": t2-t1}
            return X, info

    def get_scs_setup(self):
        """Generate the matrices and parameters to solve the sparse problem with SCS
        Constraints:
        A^T y + c = [-I  A_p ] [y_p] +  0 == 0
                    [ 0  A_tr] [y_z] +  1 == 0
                    [ 0  A_z ]       +  0 == 0
        NOTE: We build At rather than A (using dual opt form)
        NOTE: We assume here that edges correspond to the lower triangle of the affinity matrix   
        """
        # Generate clique information
        clq_sizes = []  # Size of each clique variable
        clq_start = []  # Starting indices of cliques
        ind = 0
        for clq in self.cliques:
            clq_sizes.append(len(clq))
            clq_start.append(ind)
            ind += int(len(clq)*(len(clq)+1)/2)
        # define cones
        n_pos = len(self.edges)+1
        cone = dict(l=n_pos, s=clq_sizes)
        # Adjust starting indices
        n_clique_vars = ind
        n_vars = n_clique_vars + n_pos
        # Construct Ap and objective by looping through edges in assoc graph
        b = np.zeros(n_vars)
        Ap_rows, Ap_cols, Ap_data = [], [], []
        row_ind = 0
        for edge in self.edges:
            # Loop through cliques and add to sum
            for iClq, row_c, col_c in self.get_clique_inds(edge):
                # Off diagonal multiplier
                if row_c == col_c:
                    mult = 1
                else:
                    mult = np.sqrt(2)
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(clq_sizes[iClq], row_c, col_c)
                # Add to objective
                # NOTE: Objective could also be defined on the positive cone vars
                b[vec_ind + clq_start[iClq]+n_pos] += - \
                    mult * self.affinity[edge]
                # Offset with clique starting index
                Ap_cols.append(vec_ind + clq_start[iClq])
                Ap_rows.append(row_ind)
                Ap_data.append(mult)
            # increment row of At
            row_ind += 1
        # Build Ap matrix
        Ap = sp.csc_array((Ap_data, (Ap_rows, Ap_cols)),
                          shape=(n_pos-1, n_clique_vars))
        # Construct A_tr (negative trace)
        Atr_rows, Atr_cols, Atr_data = [], [], []
        row_ind = 0
        # Loop through cliques and add to sum
        for iClq, size in enumerate(clq_sizes):
            for i in range(size):
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(size, i, i)
                # Offset with clique starting index
                Atr_cols.append(vec_ind + clq_start[iClq])
                Atr_rows.append(0)
                Atr_data.append(-1)
        Atr = sp.csc_array((Atr_data, (Atr_rows, Atr_cols)),
                           shape=(1, n_clique_vars))
        # Construct Az
        Az_rows, Az_cols, Az_data = [], [], []
        row_ind = 0
        for edge in self.fill_edges:
            # Loop through cliques and add to sum
            for iClq, row_c, col_c in self.get_clique_inds(edge):
                if row_c == col_c:
                    mult = 1
                else:
                    mult = np.sqrt(2)
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(clq_sizes[iClq], row_c, col_c)
                # Offset with clique starting index
                Az_cols.append(vec_ind + clq_start[iClq])
                Az_rows.append(row_ind)
                Az_data.append(mult)
            # increment row of At
            row_ind += 1
        # Build A_p matrix
        Az = sp.csc_array((Az_data, (Az_rows, Az_cols)),
                          shape=(row_ind, n_clique_vars))
        # Build full matrix
        I = sp.eye(n_pos, format='csc')
        Z = sp.csc_array((Az.shape[0], n_pos))
        At_top = sp.hstack([-I, sp.vstack([Ap, Atr])])
        At = sp.vstack([At_top, sp.hstack([Z, Az])])
        # Create c array
        c = np.zeros(n_pos + Az.shape[0])
        c[n_pos-1] = 1
        # Build data dict
        data = dict(A=At.T, b=b, c=c)

        return cone, data

    def get_scs_setup_homog(self):
        """Generate the matrices and parameters to solve the sparse problem with SCS
        In this case, we homogenize the problem to reduce the number of constraints.
        Constraints:
        A^T y + c = [-I    A_ineq ] [y_p  ] +  0 == 0
                    [ 0    A_eq1  ] [y_sdp] +  1 == 0
                    [ 0    A_eq2  ]         +  0 == 0
        A_p : positivity constraints on the homogenized array: u_i * h >= 0
        A_eq1: equality constraints for the trace constraint and the homogenization constraint
        A_eq2 : constraints that set the fill-in edges to zero.
        NOTE: We build At rather than A (using dual opt form)
        NOTE: We assume here that edges correspond to the lower triangle of the affinity matrix   
        """
        # Generate clique information
        clq_sizes = []  # Size of each clique variable
        clq_start = []  # Starting indices of cliques
        ind = 0
        for clq in self.cliques:
            # Add one more variable for homogenization
            clq_sizes.append(len(clq)+1)
            clq_start.append(ind)
            ind += int(clq_sizes[-1]*(clq_sizes[-1]+1)/2)
        # define cones
        n_pos = self.size
        cone = dict(l=n_pos, s=clq_sizes)
        # Adjust starting indices
        n_clique_vars = ind
        n_vars = n_clique_vars + n_pos
        # Construct Ap and objective by looping through edges in assoc graph
        b = np.zeros(n_vars)
        row_ind = 0
        for edge in self.edges:
            # Loop through cliques and add to sum
            for iClq, row_c, col_c in self.get_clique_inds(edge):
                # Off diagonal multiplier
                if row_c == col_c:
                    mult = 1
                else:
                    mult = np.sqrt(2)
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(clq_sizes[iClq], row_c, col_c)
                # Add to objective
                # NOTE: Objective could also be defined on the positive cone vars
                b[vec_ind + clq_start[iClq]+n_pos] += - \
                    mult * self.affinity[edge]

        # Generate positive constraint matrix
        A_ineq_cols, A_ineq_rows, A_ineq_data = [], [], []
        # Loop through variables. Add too positive constraint matrix
        for iVar in range(self.size):
            # Loop through cliques that "touch" this variable
            clique_indices = self.clq_lookup[iVar]
            for iClq in clique_indices:
                # Get index of variable in clique
                clq_var_ind = self.clq_var_lookups[iClq][iVar]
                # Convert to vectorized index
                vec_ind = mat2hvec_ind(
                    clq_sizes[iClq], clq_sizes[iClq]-1, clq_var_ind)
                # Get column of A_ineq. Offset with clique starting index
                col_ind = vec_ind + clq_start[iClq]
                A_ineq_cols.append(col_ind)
                A_ineq_rows.append(iVar)
                A_ineq_data.append(mult)
        # Build A_ineq matrix
        A_ineq = sp.csc_array(
            (A_ineq_data, (A_ineq_rows, A_ineq_cols)), shape=(n_pos, n_clique_vars))

        # Construct A_eq1
        Aeq1_rows, Aeq1_cols, Aeq1_data = [], [], []
        row_ind = 0
        # Loop through cliques and add to sum
        for iClq, size in enumerate(clq_sizes):
            for i in range(size-1):
                # TRACE CONSTRAINT
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(size, i, i)
                # Offset with clique starting index
                Aeq1_cols.append(vec_ind + clq_start[iClq])
                Aeq1_rows.append(0)
                Aeq1_data.append(-1)
            # HOMOGENIZING CONSTRAINT
            # Get vectorized matrix index for last element
            vec_ind = mat2hvec_ind(size, size-1, size-1)
            # Offset with clique starting index
            Aeq1_cols.append(vec_ind + clq_start[iClq])
            Aeq1_rows.append(1)
            Aeq1_data.append(-1)
        Aeq1 = sp.csc_array(
            (Aeq1_data, (Aeq1_rows, Aeq1_cols)), shape=(2, n_clique_vars))
        # Construct A_eq2
        Aeq2_rows, Aeq2_cols, Aeq2_data = [], [], []
        row_ind = 0
        for edge in self.fill_edges:
            # Loop through cliques and add to objective
            for iClq, row_c, col_c in self.get_clique_inds(edge):
                if row_c == col_c:
                    mult = 1
                else:
                    mult = np.sqrt(2)
                # Get vectorized matrix index
                vec_ind = mat2hvec_ind(clq_sizes[iClq], row_c, col_c)
                # Offset with clique starting index
                Aeq2_cols.append(vec_ind + clq_start[iClq])
                Aeq2_rows.append(row_ind)
                Aeq2_data.append(mult)
            # increment row of At
            row_ind += 1
        Aeq2 = sp.csc_array((Aeq2_data, (Aeq2_rows, Aeq2_cols)),
                            shape=(row_ind, n_clique_vars))
        # Build full matrix
        I = sp.eye(n_pos, format='csc')
        Aeq = sp.vstack([Aeq1, Aeq2])
        Z = sp.csc_array((Aeq.shape[0], n_pos))
        At_top = sp.hstack([-I, A_ineq])
        At = sp.vstack([At_top, sp.hstack([Z, Aeq])])
        # Create c array
        c = np.zeros(n_pos + Aeq.shape[0])
        c[n_pos] = 1
        c[n_pos+1] = 1
        # Build data dict
        data = dict(A=At.T, b=b, c=c)

        return cone, data

    def compute_warmstart(self, cone, data, warmstart="max-density"):
        """Compute warm start for SCS."""

        if warmstart in ["max-density", "max-clique"]:
            # Get the index of the best clique
            if warmstart == "max-density":
                vals = []
                for clique in self.cliques:
                    affinity_clique = self.affinity[np.ix_(clique, clique)]
                    vals.append(np.sum(affinity_clique.data))
                best_clique_ind = np.argmax(vals)
            elif warmstart == "max-clique":
                best_clique_ind = np.argmax(
                    [len(clique) for clique in self.cliques])

            # Generate the init SDP matrix by setting the variables of the best clique
            y_sdp = []
            for iClq, clique in enumerate(self.cliques):
                if iClq == best_clique_ind:
                    X = np.ones((len(clique), len(clique)))
                    mask = self.affinity[np.ix_(clique, clique)] > 0
                    X = X * mask
                    X = X.toarray() / len(clique)  # divide by trace to normalize
                else:
                    X = np.zeros((len(clique), len(clique)))
                # Add to list
                y_sdp.append(mat2hvec(X))
            y_sdp = np.concatenate(y_sdp)
            # Compute the other cone vars from the constraints
            n_pos = cone['l']
            A_p = data['A'].T[:n_pos, n_pos:]
            y_p = data['c'][:n_pos, None] + (A_p @ y_sdp[:, None])
            y = np.concatenate([y_p[:, 0], y_sdp])
            # Make param dict
            solver_kwargs = dict(warm_start=True,
                                 x=None,
                                 y=y,
                                 s=None)
        elif warmstart == "stored-xys":
            # Make param dict
            solver_kwargs = dict(warm_start=True, **self.stored_soln)
        elif warmstart == "stored-y":
            solver_kwargs = dict(warm_start=True,
                                 x=0*self.stored_soln['x'],
                                 y=self.stored_soln['y'],
                                 s=0*self.stored_soln['s'],
                                 )
        else:
            solver_kwargs = dict(warm_start=False,
                                 x=None,
                                 y=None,
                                 s=None)

        return solver_kwargs

    def solve_scs_sparse(self, setup_kwargs, homog=False, warmstart=None):
        """
        Solve a sparse optimization problem using the SCS solver.
        This method sets up and solves a sparse version of the optimization problem 
        using the Splitting Conic Solver (SCS). It also reconstructs the solution 
        matrix and stores relevant timing and solver information.
        Args:
            setup_kwargs (dict): Keyword arguments to configure the SCS solver setup.
            solve_kwargs (dict): Keyword arguments to configure the SCS solver execution.
        Returns:
            tuple:
                - X (numpy.ndarray): The reconstructed solution matrix.
                - info (dict): A dictionary containing solver information, including:
                    - 'time_setup': Time taken to set up the problem.
                    - 'time_solve': Time taken to solve the problem.
                    - Additional solver output from SCS.
        """
        # Set up problem
        t0 = time()
        if homog:
            cone, data = self.get_scs_setup_homog()
        else:
            cone, data = self.get_scs_setup()
        solver = scs.SCS(data, cone, **setup_kwargs)
        # Get initialization point
        if warmstart is None:
            solve_kwargs = dict()
        else:
            solve_kwargs = self.compute_warmstart(cone, data, warmstart)
        t1 = time()

        # Run Solver
        sol = solver.solve(**solve_kwargs)
        t2 = time()
        # Rebuild the solution
        if homog:
            mat_shape = (self.size+1, self.size+1)
        else:
            mat_shape = (self.size, self.size)
        X = np.zeros(mat_shape)
        start_ind = cone['l']  # Skip over positive indices
        for clique_inds in self.cliques:
            inds = clique_inds
            if homog:
                size = len(clique_inds)+1
                inds = clique_inds.copy()
                inds.append(self.size)
            else:
                size = len(clique_inds)
            vec_size = int((size+1)*size/2)
            X_c = hvec2mat(sol['y'][start_ind: start_ind+vec_size])
            X[np.ix_(inds, inds)] += X_c
            start_ind += vec_size
        # Store useful information
        info = dict(**sol)
        info['time_setup'] = t1 - t0
        info['time_solve'] = t2 - t1
        # cache solution for warmstart
        self.stored_soln = dict(x=sol['x'], y=sol['y'], s=sol['s'])
        # Add information
        info['mults'] = np.concatenate(
            [sol['s'][:cone['l']], sol['x'][cone['l']:]])
        info['ineq'] = np.zeros(data['A'].shape[1])
        info['ineq'][:cone['l']] = 1
        info['cone'] = cone
        info['data'] = data

        return X, info

    def get_dual_sol(self, info, homog=False):
        """Retrieve the dual matrix solutions"""
        # Get problem data
        cone, data = info['cone'], info['data']
        # Retrieve dual matrices
        if homog:
            mat_shape = (self.size+1, self.size+1)
        else:
            mat_shape = (self.size, self.size)
        H = np.zeros(mat_shape)
        H_c_list = []
        start_ind = cone['l']
        for clique_inds in self.cliques:
            inds = clique_inds
            if homog:
                size = len(clique_inds)+1
                inds = clique_inds.copy()
                inds.append(self.size)
            else:
                size = len(clique_inds)
            vec_size = int((size+1)*size/2)
            H_c_list.append(hvec2mat(info['s'][start_ind: start_ind+vec_size]))
            H[np.ix_(inds, inds)] = H_c_list[-1]
            start_ind += vec_size

        return H_c_list, H

    def get_clique_inds(self, edge):
        """Find all cliques that contain a given edge in the graph. Return tuple of variable indices in the form: (Clique index, row index, column index) """
        # Find the cliques that include the edge
        clique0_inds = self.clq_lookup[edge[0]]
        clique1_inds = self.clq_lookup[edge[1]]
        clique_inds = clique0_inds & clique1_inds
        # get vars across cliques
        var_list = []
        for clq_ind in clique_inds:
            ind0 = self.clq_var_lookups[clq_ind][edge[0]]
            ind1 = self.clq_var_lookups[clq_ind][edge[1]]
            # append to list, ensure that only lower triangle stored
            if ind0 >= ind1:
                var_list.append((clq_ind, ind0, ind1))
            else:
                var_list.append((clq_ind, ind1, ind0))

        return var_list

    """Diagonal Dominant Routines"""

    def get_scs_setup_dd(self):
        """Generate the matrices and parameters to solve the sparse problem with SCS.
        We solve the homogenized problem to reduce the number of constraints.
        In the initial setup, the main variable is restricted to be in the DD* cone.
        x : Relaxed PSD variable 
        s : Cone Variables
        y : Dual Cone Variables
        NOTE: We only add cost and enforce constraints on the lower triangle of the PSD matrix
        """
        # Size with homogenization
        n_h = self.size + 1
        # Get number of zero constraints (non-edges)
        n_edges = len(self.edges)
        n_not_edges = int(self.size*(self.size+1)/2) - n_edges
        n_x = int(n_h*(n_h+1)/2)
        # Get number of DD* cone constraints
        n_dd = (n_h)**2
        # Define problem cones
        # z : non-edge, trace, homogenizing constraints
        # l : DD* constraints, positivity constraits
        cone = dict(z=n_not_edges+2, l=n_dd + self.size)

        # Construct constraintmatrices
        Aeq_rows, Aeq_cols, Aeq_data = [], [], []
        c, tr_inds, pos_inds = [], [], []
        row_ind = 0
        # Loop through matrix indices
        for i in range(self.size):
            # Keep track of trace indices and positive indices (homog vector)
            tr_inds.append(mat2hvec_ind(n_h, i, i))
            pos_inds.append(mat2hvec_ind(n_h, self.size, i))
            for j in range(i, self.size):
                if self.affinity[i, j] == 0:
                    # Add constraint setting this element to zero
                    vecind = mat2hvec_ind(n_h, j, i)  # Get half vec index
                    Aeq_rows.append(row_ind)
                    Aeq_cols.append(vecind)
                    Aeq_data.append(1.0)
                    row_ind += 1  # increment row
                    # Append zero to cost
                    c.append(0)
                else:
                    # Append affinity value to cost
                    if i == j:
                        c.append(-self.affinity[i, j])
                    else:
                        c.append(-self.affinity[i, j] * np.sqrt(2))
            # last row
            c.append(0)
        # add homogenizing cost element and stack
        c.append(0)
        c = np.hstack(c)
        # Add trace constraint
        Aeq_rows += [row_ind] * len(tr_inds)
        Aeq_cols += tr_inds
        Aeq_data += [1.0] * len(tr_inds)
        row_ind += 1  # increment row
        # Add homogenization constraint
        Aeq_rows.append(row_ind)
        Aeq_cols.append(n_x-1)
        Aeq_data.append(1.0)
        row_ind += 1  # increment row
        # Check that we are at the right row
        assert row_ind == n_not_edges + \
            2, ValueError(
                "number of rows should equal number of non-edges + 2")
        # Construct Equality A Matrix
        Aeq = sp.csc_array((Aeq_data, (Aeq_rows, Aeq_cols)),
                           shape=(n_not_edges+2, n_x))
        # Construct Positive A Matrix
        rows = list(range(len(pos_inds)))
        data = [-1.0] * len(pos_inds)
        cols = pos_inds
        Apos = sp.csc_array((data, (rows, cols)), shape=(self.size, n_x))
        # Construct DD* constraints
        Add = DDConeVar.get_basis_vectors(dim=n_h).T
        assert Add.shape == (n_dd, n_x), ValueError(
            "DD star cone constraints wrong size")
        # Construct full constraint matrix
        A = sp.vstack([Aeq, Apos, -Add])
        b = np.hstack([np.zeros(n_not_edges), np.ones(2),
                      np.zeros(n_dd + self.size)])
        # Build data
        data = dict(A=A, b=b, c=c)

        return cone, data

    def solve_ddstar_cut(self, max_iter=1000, tol_eig=1e-5, num_cuts=15, scs_params={}):
        """
        Solve the problem using the diagonally dominant program with optimality cuts.
        This is the same as the column generation for outer approximation method defined in
        "Optimization over Structured Subsets of Positive Semidefinite Matrices via Column Generation"
        by Ahmadi et al.
        """
        # Get initial problem setup
        t0 = time()
        cone0, data0 = self.get_scs_setup_dd()
        cone = cone0.copy()
        data = data0.copy()
        t1 = time()

        # Tightening loop
        n_iter = 0
        solve_init = {}  # For warmstarting, empty for first iter
        while n_iter < max_iter:
            # Set up and run solver
            solver = scs.SCS(data, cone, **scs_params)
            sol = solver.solve(**solve_init)
            X = hvec2mat(sol['x'])
            # Get eigenvalues of solution
            vals, vecs = np.linalg.eigh(X)
            # Find negative eigvals
            inds = np.where(vals < -tol_eig)[0][0:num_cuts]
            n_neg_eigs = len(inds)
            if n_neg_eigs == 0:
                # If no negative eigenvalues then break
                break
            else:
                # Generate cuts corresponding to the negative eigs
                cut_rows = []
                for ind in inds:
                    evec = vecs[:, [ind]]
                    # Get new row of constraint matrix
                    cut_rows.append(
                        -mat2hvec(evec @ evec.T))
                # Update Constraints
                cone['l'] = cone['l'] + n_neg_eigs
                A_cuts = np.vstack(cut_rows)
                A_cuts_sp = sp.csc_array(A_cuts)
                A_cuts_sp.eliminate_zeros()
                data['A'] = sp.vstack([data['A'], A_cuts_sp])
                data['b'] = np.hstack([data['b'], np.zeros(n_neg_eigs)])
                # Set up warmstart for next round
                solve_init = dict(warm_start=False,
                                  x=sol['x'],
                                  s=np.hstack(
                                      [sol['s'], vals[inds]]),
                                  y=np.hstack(
                                      [sol['y'], np.ones(n_neg_eigs)]),
                                  )
            print(
                f"{n_iter}:  min eig: {vals[inds[:5]]}, opt val: {sol['info']['pobj']}")
            n_iter += 1

        # Retrieve dual solution
        V_bar = DDConeVar.get_basis_vectors(dim=self.size+1)
        n_dd = (self.size+1)**2
        alpha = sol['y'][-n_dd:][:, None]
        hvec = V_bar @ alpha
        H = hvec2mat(hvec[:, 0])

        # record cone and data
        info = dict(**sol)
        info['cone'] = cone
        info['data'] = data
        info['H'] = H

        return X, info

    """Post processing"""

    def reduce_rank(self, X, info, tol=1e-5):
        """Apply rank reduction to solution"""
        # Figure out which constraints are binding
        constraints, ineq, mults = info["constraints"], info["ineq"], info["mults"]
        binding_constraints = []
        for i, A in enumerate(constraints):
            if not ineq[i]:
                binding_constraints.append(constraints[i])
            else:
                # Check if constraint is binding
                if np.trace(A @ X) < tol:
                    binding_constraints.append(constraints[i])

        V = rank_reduction(binding_constraints, X, null_tol=tol)
        return V

    def process_sdp_var(self, X, homog=False):
        """Post process SDP variable into an actual solution."""
        # Decompose SDP solution
        evals, evecs = np.linalg.eigh(X)
        er = evals[-1] / evals[-2]
        x_opt = evecs[:, -1] * np.sqrt(evals[-1])

        if homog:
            if x_opt[-1] < 0:
                x_sol = -x_opt[:-1]
            else:
                x_sol = x_opt[:-1]
        else:
            x_sol = x_opt
        # Select inliers
        thresh = np.max(x_sol) / 2
        inliers = (x_sol > thresh).astype(float)

        return inliers, er, x_opt

# The vec function as documented in api/cones


def mat2hvec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


# The mat function as documented in api/cones
def hvec2mat(s):
    """Convert matrix in half vectorized form into a symmetric matrix
    Assume that the vector has more columns than rows.
    Column major ordering assumed.
    """
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


def vec2hvec(n):
    """convert full vec to scaled half vec"""
    rows, cols, vals = [], [], []
    col_ind = 0
    for j in range(n):  # Column Loop
        for i in range(n):  # Row Loop
            if i == j:
                rows.append(mat2hvec_ind(n, i, j))
                cols.append(col_ind)
                vals.append(1.0)
            elif i > j:
                rows.append(mat2hvec_ind(n, i, j))
                cols.append(col_ind)
                vals.append(np.sqrt(2))
            col_ind += 1
    return sp.csc_array((vals, (rows, cols)), shape=(int(n*(n+1)/2), n*n))


def mat2hvec_ind(n, row, col):
    """convert SDP matrix indices to index of the half vectorization
    Column major ordering assumed"""
    assert row >= col, ValueError("Lower triangular indices are assumed")
    return int(n*col - (col-1)*col/2 + row - col)


def mat2vec_ind(n, row, col):
    """Convert matrix indices to vectorized index. Column major ordering assumed"""
    return n*col+row


def mat_fusion(X):
    """Convert sparse matrix X to fusion format"""
    try:
        X.eliminate_zeros()
    except AttributeError:
        X = sp.csr_array(X)
    I, J = X.nonzero()
    V = np.array(X[I, J]).flatten().astype(np.double)
    I = I.astype(np.int32)
    J = J.astype(np.int32)
    return fu.Matrix.sparse(*X.shape, I, J, V)


def cvxmat2sparse(X: spmatrix):
    rows = np.array(X.I)[:, 0]
    cols = np.array(X.J)[:, 0]
    vals = np.array(X.V)[:, 0]
    return sp.csc_array((vals, (rows, cols)), X.size)


def randsphere(m, n, r):
    """Draw random points from within a sphere."""
    X = np.random.randn(m, n)
    s2 = np.sum(X**2, axis=1)
    X = X * np.tile((r*(gammainc(n/2, s2/2)**(1/n)) /
                    np.sqrt(s2)).reshape(-1, 1), (1, n))
    return X


def generate_bunny_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21):
    """Generate a dataset for the registration problem.

    Parameters
    ----------
    pcfile : str
        Path to the point cloud file.
    m : int
        Total number of associations in the problem.
    n1 : int
        Number of points used on model (i.e., seen in view 1).
    n2o : int
        Number of outliers in data (i.e., seen in view 2).
    outrat : float
        Outlier ratio of initial association set.
    sigma : float
        Uniform noise [m] range.
    T_21 : np.ndarray
        Ground truth transformation from view 1 to view 2.

        Returns
        -------
        D1 : np.ndarray
            Model points in view 1.
        D2 : np.ndarray
            Data points in view 2.
        Agt : np.ndarray
            Ground truth associations.
        A : np.ndarray
            Initial association set.
    """
    pcd = o3d.io.read_point_cloud(pcfile)

    n2 = n1 + n2o  # number of points in view 2
    noa = round(m * outrat)  # number of outlier associations
    nia = m - noa  # number of inlier associations

    if nia > n1:
        raise ValueError("Cannot have more inlier associations "
                         "than there are model points. Increase"
                         "the number of points to sample from the"
                         "original point cloud model.")

    # Downsample from the original point cloud, sample randomly
    I = np.random.choice(len(pcd.points), n1, replace=False)
    D1 = np.asarray(pcd.points)[I, :].T

    # Rotate into view 2 using ground truth transformation
    D2 = T_21[0:3, 0:3] @ D1 + T_21[0:3, 3].reshape(-1, 1)
    # Add noise uniformly sampled from a sigma cube around the true point
    eta = np.random.uniform(low=-sigma/2., high=sigma/2., size=D2.shape)
    # Add noise to view 2
    D2 += eta

    # Add outliers to view 2
    R = 1  # Radius of sphere
    O2 = randsphere(n2o, 3, R).T + D2.mean(axis=1).reshape(-1, 1)
    D2 = np.hstack((D2, O2))

    # Correct associations to draw from
    # NOTE: These are the exact correponsdences between views
    Agood = np.tile(np.arange(n1).reshape(-1, 1), (1, 2))

    # Incorrect association to draw from
    # NOTE: Picks any other correspondence than the correct one
    Abad = np.zeros((n1*n2 - n1, 2))
    itr = 0
    for i in range(n1):
        for j in range(n2):
            if i == j:
                continue
            Abad[itr, :] = [i, j]
            itr += 1

    # Sample good and bad associations to satisfy total
    # num of associations with the requested outlier ratio
    IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
    IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
    A = np.concatenate((Agood[IAgood, :], Abad[IAbad, :])).astype(np.int32)

    # Ground truth associations
    Agt = Agood[IAgood, :]

    return (D1, D2, Agt, A)


def get_err(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(
        np.arccos(min(max(((Terr[0:3, 0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3, 3])
    return (rerr, terr)


def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def get_affinity_from_points(points_1, points_2, associations, threshold=0.5):
    # Define invariant function
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.sigma = 0.01
    iparams.epsilon = 0.02
    invariant = clipperpy.invariants.EuclideanDistance(iparams)
    # Define rounding strategy
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    # define clipper object
    clipper = clipperpy.CLIPPER(invariant, params)
    
    # Get pairwise consistency matrix
    clipper.score_pairwise_consistency(points_1, points_2, associations)
    # Get affinity
    M = clipper.get_affinity_matrix()
    # HACK Manual threshold
    if threshold > 0.0:
        M = (M > threshold).astype(float)
        # Set constraint and affinity matrix to thresholded values.
        clipper.set_matrix_data(M=M, C=M)
    # Convert to sparse
    M = sp.csr_array(M)
    M.eliminate_zeros()    
    
    return M, clipper


def prune_affinity(affinity, clique_size_lb):
    """Prune the graph based on the fact that nodes with degree lower than the max clique size can be safely removed.
    NOTE: It is assumed that the affinity matrix is binary (i.e., thresholded)
    """
    # Get number of nodes
    n = affinity.shape[0]
    # Get edges
    edges = list(zip(*affinity.nonzero()))
    edges = [edge for edge in edges if edge[1] > edge[0]]
    # Define graph structure
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    G.add_edges_from(edges)
    # Get k-core
    G_pruned = nx.k_core(G, k=clique_size_lb-1)
    keep_inds =list(G_pruned.nodes)
    
    return affinity[keep_inds][:, keep_inds]
    

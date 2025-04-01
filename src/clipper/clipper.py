import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from scipy.special import gammainc
import matplotlib.pyplot as plt
import cvxpy as cp
import mosek.fusion as fu
import scipy.sparse as sp
import clipperpy
from cvxopt import amd, spmatrix
import chompack
from time import time




class ConsistencyGraphProb():
    def __init__(self, affinity):
        # Init affinity matrix
        if sp.issparse(affinity):
            self.affinity = affinity
        else:
            self.affinity = sp.csc_array(affinity)
        
        self.size = self.affinity.shape[0]
        # Run Symbolic Factorization
        self.symb_fact_affinity()
        
        # Mosek options
        TOL = 1e-7
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
    
    
        
    def get_affine_constraints(self):
        """Generate all affine constraints for the problem
        Equalities assumed to be of form: <A, X> + b = 0 
        Inequalities assumed to be of form: <A, X> + b >= 0"""
        M = self.affinity
        n = M.shape[0]
        eqs = []
        ineqs = []
        for i in range(n):
            for j in range(i,n):
                # Construct selection matrix
                if i == j:
                    vals = [1]
                    rows = [i]
                    cols= rows
                else:
                    vals = [0.5, 0.5]
                    rows = [i, j]
                    cols = [j, i]
                A = sp.csr_array((vals, (rows,cols)), shape=(n,n))
                # Each element of X is either zero or non-negative, depending on M
                if M[i,j] == 0:
                    eqs.append((A, 0.0))
                else:
                    ineqs.append((A, 0.0))
        # Add trace constraint
        A = sp.identity(n,format="csr")
        ineqs.append((-A, 1.0))

        return eqs, ineqs        
    
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

    def symb_fact_affinity(self, merge_function=None):
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
        self.edges = [tuple(sorted([rows[i], cols[i]])) for i in range(len(rows)) if rows[i]>=cols[i]]
        self.pattern = spmatrix(1.0, rows, cols, self.affinity.shape)
        # get information from factorization
        if merge_function is None:
            merge_function = self.merge_cosmo
        self.symb = chompack.symbolic(self.pattern, p=amd.order, merge_function=merge_function)
        self.cliques = self.symb.cliques()
        self.sepsets = self.symb.separators()
        self.parents = self.symb.parent()
        # Get variable list in permuted order (symbolic factorization reorders things)
        var_list = list(range(self.size))
        var_list_perm = [var_list[p] for p in self.symb.p]
        self.ind_to_clq=dict()
        for iClq, clique in enumerate(self.cliques):
            # Get the cliques and separator sets in the original ordering
            self.cliques[iClq] = [var_list_perm[v] for v in clique]
            self.sepsets[iClq] = set([var_list_perm[v] for v in self.sepsets[iClq]])
            # Generate mapping from index to clique mapping as a dictionary of lists
            for ind in self.cliques[iClq]:
                if ind not in self.ind_to_clq.keys():
                    self.ind_to_clq[ind] = set([iClq])
                else:
                    self.ind_to_clq[ind].add(iClq)
        # Get a list of "fill-in" edges
        pattern_filled = self.symb.sparsity_pattern(reordered=False)
        fillin = pattern_filled - self.pattern
        self.fill_edges = []
        self.fill_pattern = cvxmat2sparse(fillin)
        self.fill_pattern.eliminate_zeros()
        # NOTE: only store edges on lower triangle to avoid double counting
        self.fill_edges = [tuple(sorted(edge)) for edge in zip(*self.fill_pattern.nonzero()) if edge[0] >= edge[1]]
        
        # Keep filed pattern
        self.filled_pattern = cvxmat2sparse(pattern_filled)
        self.filled_pattern.eliminate_zeros()
        
        
            
    def solve_cvxpy(self, options=None, verbose = False):
        """Solve the MSRC problem with CVXPY"""
        if options is None:
            options = self.options_cvxpy
        
        # Get the affinity matrix in sparse format
        M = self.get_affinity_matrix()
        n = M.shape[0]
        if verbose:
            print("Constructing CVXPY problem")
        # Define vars
        X = cp.Variable(M.shape, symmetric=True)
        # PSD Constraint
        constraints = [X >> 0]
        # Add affine constraints   
        eqs, ineqs= self.get_affine_constraints()
        for A,b in eqs:
            constraints.append(cp.trace(A @ X) + b == 0)
        for A,b in ineqs:    
            constraints.append(cp.trace(A @ X) + b >= 0)
            
        if verbose:
            print("Solving...")
            options["verbose"] = verbose
        cprob = cp.Problem(cp.Maximize(cp.trace(M @ X)), constraints)
        try:
            cprob.solve(
                solver="MOSEK",
                **options,
            )
        except cp.SolverError as e:
            cost = None
            X = None
            H = None
            yvals = None
            msg = f"infeasible / unknown: {e}"
        else:
            if np.isfinite(cprob.value):
                cost = cprob.value
                X = X.value
                H = constraints[0].dual_value
                yvals = [c.dual_value for c in constraints[1:]]
                msg = "converged"
            else:
                cost = None
                X = None
                H = None
                yvals = None
                msg = "unbounded"
                
        return (X, H, yvals)
    
    def solve_fusion(self, options=None, verbose = False):
        """Solve the MSRC problem with Mosek Fusion"""
        if options is None:
            options = self.options_fusion
        
        n = self.size
        M = self.affinity
        
        with fu.Model("dual") as mdl:
            if verbose:
                print("Constructing Problem")
            # Create SDP variable
            X = mdl.variable("X", fu.Domain.inPSDCone(n))
            # Add constriants
            for i in range(n):
                for j in range(i,n):
                    # Construct selection matrix
                    if i == j:
                        vals = [1]
                        rows = [i]
                        cols= rows
                    else:
                        vals = [0.5, 0.5]
                        rows = [i, j]
                        cols = [j, i]
                    A = fu.Matrix.sparse(n, n, rows, cols, vals)
                    # Each element of X is either zero or non-negative, depending on M
                    if M[i,j] == 0:
                        mdl.constraint(fu.Expr.dot(A, X), fu.Domain.equalsTo(0))
                    else:
                        mdl.constraint(fu.Expr.dot(A, X), fu.Domain.greaterThan(0))
            # Add Trace Constraint
            # NOTE: Ineq below should be interchangeable with 
            A = mat_fusion(sp.identity(n, format='csr'))
            mdl.constraint(fu.Expr.dot(A, X), fu.Domain.lessThan(1))
            # mdl.constraint(fu.Expr.dot(A, X), fu.Domain.equalsTo(1))
            
            # Add affinity matrix objective
            mdl.objective(fu.ObjectiveSense.Maximize, fu.Expr.dot(mat_fusion(M), X))

            if verbose:
                mdl.setLogHandler(sys.stdout)
                # mdl.writeTask("problem.ptf")

            for key, val in options.items():
                mdl.setSolverParam(key, val)

            mdl.acceptedSolutionStatus(fu.AccSolutionStatus.Anything)
            mdl.solve()

            if mdl.getProblemStatus() in [
                fu.ProblemStatus.PrimalAndDualFeasible,
                fu.ProblemStatus.Unknown,
            ]:
                cost = mdl.primalObjValue()
                H = np.reshape(X.dual(), (n,n))
                X = np.reshape(X.level(), (n,n))
                msg = f"success with status {mdl.getProblemStatus()}"
                success = True
            else:
                cost = None
                H = None
                X = None
                msg = f"solver failed with status {mdl.getProblemStatus()}"
                success = False
            info = {"success": success, "cost": cost, "msg": msg, "H": H}
            return X, info
    

    def solve_fusion_sparse(self, options=None, verbose = False):
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
                    obj_list.append(fu.Expr.mul(2*self.affinity[edge], clq_sum))
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
                    X[np.ix_(clique_inds, clique_inds)] += np.reshape(X_cs[iClq].level(), (n,n))
                cost = mdl.primalObjValue()
                msg = f"success with status {mdl.getProblemStatus()}"
                success = True
            else:
                cost = None
                X = None
                msg = f"solver failed with status {mdl.getProblemStatus()}"
                success = False
            info = {"success": success, "cost": cost, "msg": msg, "time_setup":t1-t0,"time_solve":t2-t1}
            return X, info
    
    def get_clique_inds(self, edge):
        """Find all cliques that contain a given edge in the graph. Return tuple of variable indices in the form: (Clique index, row index, column index) """
        # Find the cliques that include the edge
        clique0_inds = self.ind_to_clq[edge[0]]
        clique1_inds = self.ind_to_clq[edge[1]]
        clique_inds = clique0_inds & clique1_inds
        # Combine vars across cliques
        var_list = []
        for clq_ind in clique_inds:
            clique = self.cliques[clq_ind]
            ind0 = clique.index(edge[0])
            ind1 = clique.index(edge[1])
            var_list.append((clq_ind, ind0, ind1))
        
        return var_list
        
    
    def process_sdp_var(self, X):
        # Decompose SDP solution
        evals, evecs = np.linalg.eigh(X)
        er = evals[-1] / evals[-2]
        x_opt = evecs[:,-1] * np.sqrt(evals[-1])
        
        # Select inliers
        thresh = np.max(x_opt) / 2
        inliers = (x_opt > thresh).astype(float)
        
        return inliers, er, x_opt

def mat_fusion(X):
    """Convert sparse matrix X to fusion format"""
    try:
        X.eliminate_zeros()
    except AttributeError:
        X = sp.csr_array(X)
    I, J = X.nonzero()
    I = I.astype(np.int32)
    J = J.astype(np.int32)
    V = X.data.astype(np.double)
    return fu.Matrix.sparse(*X.shape, I, J, V)

def cvxmat2sparse(X:spmatrix):
    rows = np.array(X.I)[:,0]
    cols = np.array(X.J)[:,0]
    vals = np.array(X.V)[:,0]
    return sp.csc_array((vals, (rows, cols)), X.size)

def randsphere(m,n,r):
    """Draw random points from within a sphere."""
    X = np.random.randn(m, n)
    s2 = np.sum(X**2, axis=1)
    X = X * np.tile((r*(gammainc(n/2,s2/2)**(1/n)) / np.sqrt(s2)).reshape(-1,1),(1,n))
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

        n2 = n1 + n2o # number of points in view 2
        noa = round(m * outrat) # number of outlier associations
        nia = m - noa # number of inlier associations

        if nia > n1:
            raise ValueError("Cannot have more inlier associations "
                            "than there are model points. Increase"
                            "the number of points to sample from the"
                            "original point cloud model.")

        # Downsample from the original point cloud, sample randomly
        I = np.random.choice(len(pcd.points), n1, replace=False)
        D1 = np.asarray(pcd.points)[I,:].T

        # Rotate into view 2 using ground truth transformation
        D2 = T_21[0:3,0:3] @ D1 + T_21[0:3,3].reshape(-1,1)
        # Add noise uniformly sampled from a sigma cube around the true point
        eta = np.random.uniform(low=-sigma/2., high=sigma/2., size=D2.shape)
        # Add noise to view 2
        D2 += eta

        # Add outliers to view 2
        R = 1 # Radius of sphere
        O2 = randsphere(n2o,3,R).T + D2.mean(axis=1).reshape(-1,1)
        D2 = np.hstack((D2,O2))

        # Correct associations to draw from
        # NOTE: These are the exact correponsdences between views
        Agood = np.tile(np.arange(n1).reshape(-1,1),(1,2))

        # Incorrect association to draw from
        #NOTE: Picks any other correspondence than the correct one
        Abad = np.zeros((n1*n2 - n1, 2))
        itr = 0
        for i in range(n1):
            for j in range(n2):
                if i == j:
                    continue
                Abad[itr,:] = [i, j]
                itr += 1

        # Sample good and bad associations to satisfy total
        # num of associations with the requested outlier ratio
        IAgood = np.random.choice(Agood.shape[0], nia, replace=False)
        IAbad = np.random.choice(Abad.shape[0], noa, replace=False)
        A = np.concatenate((Agood[IAgood,:],Abad[IAbad,:])).astype(np.int32)

        # Ground truth associations
        Agt = Agood[IAgood,:]
        
        return (D1, D2, Agt, A)

def get_err(T, That):
    Terr = np.linalg.inv(T) @ That
    rerr = abs(np.arccos(min(max(((Terr[0:3,0:3]).trace() - 1) / 2, -1.0), 1.0)))
    terr = np.linalg.norm(Terr[0:3,3])
    return (rerr, terr)

def draw_registration_result(source, target, transformation):
    import copy
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def get_affinity_from_points(points_1, points_2, associations):
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
    M = sp.csr_array(M)
    M.eliminate_zeros()
    return M, clipper
        
    
    
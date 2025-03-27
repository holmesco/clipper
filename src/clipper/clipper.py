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



class ConsistencyGraphProb():
    def __init__(self, points_1, points_2, associations):
        self.points_1 = points_1
        self.points_2 = points_2
        self.associations = associations
        # Define invariant function    
        iparams = clipperpy.invariants.EuclideanDistanceParams()
        iparams.sigma = 0.01
        iparams.epsilon = 0.02
        invariant = clipperpy.invariants.EuclideanDistance(iparams)
        # Define rounding strategy
        params = clipperpy.Params()
        params.rounding = clipperpy.Rounding.DSD_HEU
        # define clipper object
        self.clipper = clipperpy.CLIPPER(invariant, params)
        # Get pairwise consistency matrix
        self.clipper.score_pairwise_consistency(points_1, points_2, associations)

        # Init affinity matrix
        self.affinity=self.get_affinity_matrix()
        
        # Mosek options
        TOL = 1e-10
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
    
    def get_affinity_matrix(self):
        M = self.clipper.get_affinity_matrix()
        M = sp.csr_array(M)
        M.eliminate_zeros()
        return M
        
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
    
        
    def get_certificate(self, x_cand):
        """Attept to generate a certificate for this problem"""
        # Get cost matrix
        M = self.affinity
        # Get Constraints
        eqs, ineqs = self.get_affine_constraints()
        
        
        
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
        
        # Get the affinity matrix in sparse format
        if verbose:
            print("Retrieving affinity matrix")
        M = self.clipper.get_affinity_matrix()
        M = sp.csr_array(M)
        M.eliminate_zeros()
        n = M.shape[0]
        
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
        
    def process_sdp_var(self, X):
        # Decompose SDP solution
        evals, evecs = np.linalg.eigh(X)
        er = evals[-1] / evals[-2] > 1e6
        x_opt = evecs[:,-1] * np.sqrt(evals[-1])
        
        # Select inliers
        thresh = np.max(x_opt) / 2
        inliers = x_opt > thresh
        
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

def randsphere(m,n,r):
    """Draw random points from within a sphere."""
    X = np.random.randn(m, n)
    s2 = np.sum(X**2, axis=1)
    X = X * np.tile((r*(gammainc(n/2,s2/2)**(1/n)) / np.sqrt(s2)).reshape(-1,1),(1,n))
    return X
        
def generate_dataset(pcfile, m, n1, n2o, outrat, sigma, T_21):
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
    

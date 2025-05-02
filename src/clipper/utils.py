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


def reduce_rank(X, info, tol=1e-5):
    """Apply rank reduction to solution"""
    # Figure out which constraints are binding
    constraints, ineq = info["constraints"], info["ineq"]
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


def vec2hvec(n, rescale=True):
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

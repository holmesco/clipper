import matplotlib.pyplot as plt
import mosek.fusion as mf
import numpy as np
import scipy.sparse as sp
from diffcp.cones import unvec_symm, vec_symm
from scipy.sparse.linalg import svds
import mosek.fusion as fu



def rank_reduction(
    Constraints,
    X_hr,
    rank_tol=1e-5,
    null_tol=1e-5,
    eig_tol=1e-9,
    null_method="svd",
    targ_rank=None,
    max_iter=None,
    verbose=False,
):
    """Algorithm that searches for a low rank solution to the SDP problem, given an existing high rank solution.
    Based on the algorithm proposed in "Low-Rank Semidefinite Programming:Theory and Applications by Lemon et al.


    """
    # Get initial low rank factor
    V = get_low_rank_factor(X_hr, rank_tol)
    r = V.shape[1]
    if verbose:
        print(f"Initial rank of solution: {r}")
    # Get constraint operator matrix
    vAv = get_constraint_op(Constraints, V)

    # REDUCE RANK
    n_iter = 0
    while (max_iter is None or n_iter < max_iter) and (
        targ_rank is None or r > targ_rank
    ):
        # Compute null space
        vec, s_min = get_min_sing_vec(vAv, method=null_method)
        if targ_rank is None and s_min > null_tol:
            if verbose:
                print("Null space has no dimension. Exiting.")
            break
        # Get basis vector corresponding to the lowest gain eigenvalue (closest to nullspace)
        Delta = unvec_symm(vec, dim=V.shape[1])
        # Compute Eigenspace of Delta
        lambdas, Q = np.linalg.eigh(Delta)
        # find max magnitude eigenvalue
        indmax = np.argmax(np.abs(lambdas))
        max_lambda = lambdas[indmax]
        # Compute reduced lambdas
        alpha = -1 / max_lambda
        lambdas_red = 1 + alpha * lambdas
        # Check which eigenvalues are still nonzero
        inds = lambdas_red > eig_tol
        # Get update matrix
        Q_tilde = Q[:, inds] * np.sqrt(lambdas_red[inds])
        # Update Nullspace matrix
        vAv = update_constraint_op(vAv, Q_tilde, dim=r)
        # Update Factor
        V = V @ Q_tilde
        r = V.shape[1]
        n_iter += 1

        if verbose:
            print(f"iter: {n_iter}, min s-value: {s_min}, rank: {r}")

    return V



def get_full_constraint_op(Constraints, C):
    """Function to compute the constraint operator whose nullspace characterizes the optimal solution.
    Computes the full operator rather than a subspace."""
    A_bar = []
    for A, b in Constraints:
        A_bar.append(A.tocoo().reshape((1, -1)))
    A_bar.append(C.tocoo().reshape((1, -1)))
    A_bar = sp.vstack(A_bar)
    return A_bar


def nullspace_projection(A, x, method="direct"):
    """Solve the nullspace projection for a large sparse matrix. That is we want to find:
    x_p = (I - A^T (A A^T)^(-1) A) x = x - (A^+) A x

    We solve this by defining y = A x, then solving A z = y as a least squares problem
    """

    if method == "cg":
        Ax = A @ x
        # construct normal eq matrix
        AAt = A @ A.T
        z, info = sp.linalg.cg(AAt, Ax)
        # Projection
        x_proj = x - A.T @ z[:, None]
    elif method == "lsqr":
        Ax = A @ x
        output = sp.linalg.lsqr(A, Ax)
        z = output[0]
        x_proj = x - z[:, None]
    elif method == "direct":
        Ax = A @ x
        # construct normal eq matrix
        AAt = A @ A.T
        z = np.linalg.solve(AAt.toarray(), Ax)
        # Projection
        x_proj = x - A.T @ z
    return x_proj


def get_min_sing_vec(A, method="svd"):
    """Get right singular vectors associated with minimum singular value."""
    if method == "svds":
        # Get minimum singular vector
        # NOTE: This method is fraught with numerical issues, but should be faster than computing all of the singular values
        s_min, vec = svds(A, k=1, which="SM")

    elif method == "svd":
        # Get all (right) singular vectors (descending order)
        U, S, Vh = np.linalg.svd(A, full_matrices=True)
        if len(S) < A.shape[1]:
            s_min = 0
        else:
            s_min = S[-1]
        vec = Vh[-1, :]
    else:
        raise ValueError("Singular vector method unknown")

    return vec, s_min


def get_low_rank_factor(X, rank_tol=1e-6, rank=None):
    """Get the low rank factorization of PSD matrix X. Tolerance is relative"""
    # get eigenspace
    vals, vecs = np.linalg.eigh(X)
    # remove zero eigenspace
    val_max = np.max(vals)
    if rank is None:
        rank = np.sum(vals > rank_tol * val_max)
    n = X.shape[0]
    V = vecs[:, (n - rank) :] * np.sqrt(vals[(n - rank) :])
    return V


def get_reduced_constraints(Constraints, V):
    reduced_constraints = []
    for A, b in Constraints:
        reduced_constraints.append((V.T @ A @ V, b))
    return reduced_constraints


def get_constraint_op(Constraints, V):
    """Function to compute the constraint operator whose nullspace characterizes the optimal solution."""
    Av = []
    for A in Constraints:
        Av.append(vec_symm(V.T @ A @ V))
    Av = np.stack(Av)
    return Av


def update_constraint_op(Av, Q_tilde, dim):
    """Update the nullspace matrix. Updating this way is cheaper than reproducing the matrix, because it is performed in the lower dimension."""
    Av_updated = []
    for i, row in enumerate(Av):
        A = unvec_symm(row, dim=dim)
        Av_updated.append(vec_symm(Q_tilde.T @ A @ Q_tilde))
    Av_updated = np.stack(Av_updated)
    return Av_updated

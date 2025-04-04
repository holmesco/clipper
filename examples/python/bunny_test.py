import os
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from src.clipper.clipper import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points
from sksparse import cholmod
from pandas import DataFrame
import seaborn as sns

PARAMS_SCS_DFLT = dict(max_iters = 2000,
                      acceleration_interval = 10,
                      acceleration_lookback= 10,
                      eps_abs = 1e-3,
                      eps_rel = 1e-3,
                      eps_infeas=1e-7,
                      time_limit_secs=0,
                      verbose = False)

class BunnyProb():
    def __init__(self, m = 100, n1 = 100, n2o = 10, outrat = 0.9, sigma = 0.01, threshold=0, seed=0):
        self.outrat = outrat
        # Set up common variables for tests
        pcfile = 'examples/data/bun10k.ply'
        T_21 = np.eye(4)
        T_21[0:3, 0:3] = Rotation.random().as_matrix()
        T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))
        # Generate dataset
        np.random.seed(seed)
        D1, D2, Agt, A = generate_bunny_dataset(
            pcfile, m, n1, n2o, outrat, sigma, T_21
        )     
        # Generate affinity
        self.affinity, self.clipper = get_affinity_from_points(D1, D2, A) 
            # Generate a solution vector
        x = np.zeros((A.shape[0], 1))
        for i,a in enumerate( A):
            # Find any associations that are in the GT set
            if np.any(np.sum(a == Agt,axis=1) == 2):
                x[i] = 1
            else:
                x[i] = 0
        self.x = x
        # Set up python version of the problem
        if threshold > 0:
            affinity = threshold_affinity(self.affinity)
        else:
            affinity = self.affinity
        self.cgraph = ConsistencyGraphProb(affinity=affinity)

        
    def solve_clipper(self):
        """Solve SDP using CLIPPER formulation"""
        t0 = time()
        self.clipper.solve_as_msrc_sdr()
        t1 = time()
        
        soln = self.clipper.get_solution()
        inliers = np.zeros(self.cgraph.size)
        inliers[soln.nodes] = 1
        
        return inliers, t1-t0
    
    def solve_fusion_sparse(self, verbose=False):
        X, info = self.cgraph.solve_fusion_sparse(verbose=verbose)
        inliers, er, x_opt = self.cgraph.process_sdp_var(X)
        return inliers, info['time_setup'], info['time_solve']
    
    def solve_fusion(self, verbose=False):
        X, info = self.cgraph.solve_fusion(verbose=verbose)
        inliers, er, x_opt = self.cgraph.process_sdp_var(X)
        return inliers, info['time_setup'], info['time_solve']
    
    def solve_scs_sparse(self, setup_kwargs, warmstart):
        X, info = self.cgraph.solve_scs_sparse(setup_kwargs, warmstart)
        inliers, er, x_opt = self.cgraph.process_sdp_var(X)
        return inliers, info['time_setup'], info['time_solve']

def threshold_affinity(M, thresh = 0.5):
    """Threshold the affinity matrix"""
    cols_thrsh, rows_thrsh = [], []
    rows, cols = M.nonzero()
    vals = M.data
    for i in range(len(cols)):
        if vals[i] >= thresh:
            cols_thrsh.append(cols[i])
            rows_thrsh.append(rows[i])
        
    M_thrsh = sp.sparse.csc_array((np.ones(len(cols_thrsh)),(rows_thrsh, cols_thrsh)), shape=M.shape)
    return M_thrsh

def study_graph_decomp(mult=2):
    # Get setup
    m = int(100*mult)
    n1 = int(100*mult)
    n2o = int(10*mult)
    prob = BunnyProb(m=m, n1=n1, n2o=n2o)
    
    M = prob.affinity.toarray()
    deg = []
    for row in range(M.shape[0]):
        deg.append(np.sum(M[row,row:] > 0))
    deg = np.array(deg)
    inliers = prob.x[:,0] == 1
    association = np.array(range(M.shape[0]))
    plt.plot(association[inliers], deg[inliers], '.b', label="inliers")
    plt.plot(association[~inliers], deg[~inliers], '.r', label='outliers')
    plt.ylabel("Degree")
    plt.xlabel("Association")
    plt.legend()
    
    df = DataFrame(dict(degree=deg, inliers=inliers, association=association))
    plt.figure()
    sns.histplot(df[inliers == True], x='degree',label='Inliers')
    sns.histplot(df[inliers == False], x='degree',label='Outliers')
    plt.legend()
    plt.show()
    
    # Check different elimination orderings
    methods = ['natural', 'amd', 'metis', 'nesdis', 'colamd', 'default','best']
    
    fillin = []
    for method in methods:   
        order = cholmod.analyze(prob.affinity, ordering_method=method).P()
        prob.cgraph.symb_fact_affinity(order=order)
        fillin.append(len(prob.cgraph.fill_edges))
        print(f"{method}:  {fillin[-1]}")
    
    

def run_scs(mult=2, warmstart=None):
    m = int(100*mult)
    n1 = int(100*mult)
    n2o = int(10*mult)
    prob = BunnyProb(m=m, n1=n1, n2o=n2o)
    
    # Set up SCS options
    scs_params = PARAMS_SCS_DFLT
    scs_params['verbose']=True
    # Sparse solver    
    inliers_s, t_setup_s, t_solve_s = prob.solve_scs_sparse(scs_params, warmstart)
    
    print(f"Runtime: {t_solve_s}")
 
def speed_compare(mult = 2):
    m = int(100*mult)
    n1 = int(100*mult)
    n2o = int(10*mult)
    prob = BunnyProb(m=m, n1=n1, n2o=n2o)
    
    # Set up SCS options
    scs_params = PARAMS_SCS_DFLT
    # Enable multithreading
    os.environ["OMP_NUM_THREADS"] = "30"  # Set to the number of desired threads
    # Sparse solver    
    inliers_s, t_setup_s, t_solve_s = prob.solve_scs_sparse(**scs_params)
    # Non-sparse solver
    inliers_c, time_c = prob.solve_clipper()
    # Check difference between solutions
    n_diff = np.sum(np.abs(inliers_c - inliers_s))
    
    s=f"""
    Number of Associations: {m}
    Number of inlier differences: {n_diff}
    SCS Sparse:
    Setup time: {t_setup_s}, Solve time {t_solve_s}
    SCS Non-Sparse (CLIPPER):
    Time: {time_c}
    """
    print(s)
 
if __name__ == "__main__":
    # run_scs(mult=5)
    # run_scs(mult=5, warmstart="max-density")
    study_graph_decomp()
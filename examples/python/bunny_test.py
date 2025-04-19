import os
import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from src.clipper.clipper import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points, PARAMS_SCS_DFLT
from sksparse import cholmod
import pandas as pd
import seaborn as sns

class BunnyProb():
    def __init__(self, m = 100, n1 = 100, n2o = 10, outrat = 0.9, sigma = 0.01, threshold=None, seed=0):
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
        self.cgraph = ConsistencyGraphProb(affinity=self.affinity, threshold=threshold)

    def get_prec_recall(self, inliers):
        # Report information
        true_pos = np.sum(self.x * inliers)
        all_pos = np.sum(inliers)
        all_true_pos = np.sum(self.x)
        precision = true_pos / all_pos
        recall = true_pos /  all_true_pos 
    
        return precision, recall
        
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
    
    def solve_scs_sparse(self, setup_kwargs, warmstart):
        X, info = self.cgraph.solve_scs_sparse(setup_kwargs, warmstart)
        inliers, er, x_opt = self.cgraph.process_sdp_var(X)
        
        return inliers, info['time_setup'], info['time_solve']


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
    
    df = pd.DataFrame(dict(degree=deg, inliers=inliers, association=association))
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
    
    
def test_warmstart_scs(mult=2):
    m = int(100*mult)
    n1 = int(100*mult)
    n2o = int(10*mult)
    prob = BunnyProb(m=m, n1=n1, n2o=n2o)
    
    # Set up SCS options
    scs_params = PARAMS_SCS_DFLT
    scs_params['verbose']=True
    # Sparse solver    
    _, _, t_solve_cold = prob.solve_scs_sparse(scs_params,warmstart=None)
    # warmstart (full solution)
    _, _, t_solve_warm = prob.solve_scs_sparse(scs_params, warmstart = "stored-xys")
    # warmstart (just cone)
    _, _, t_solve_warmcone = prob.solve_scs_sparse(scs_params, warmstart = "stored-y")
    # Results
    print(f"Runtime no warmstart: {t_solve_cold}")
    print(f"Runtime full warmstart: {t_solve_warm}")
    print(f"Runtime just cone: {t_solve_warmcone}")
    
def run_fusion_v2(mult=2, verbose = False):
    m = int(100*mult)
    n1 = int(100*mult)
    n2o = int(10*mult)
    prob = BunnyProb(m=m, n1=n1, n2o=n2o)
    
    # Run fusion solver    
    X, info = prob.cgraph.solve_fusion_v2(verbose=verbose)
    inliers, er, x_opt = prob.cgraph.process_sdp_var(X)
    
    precision, recall = prob.get_prec_recall(inliers)
    
    output = dict(
        inliers=inliers,
        er=er,
        t_setup=info['time_setup'],
        t_solve=info['time_solve'],
        precision=precision,
        recall=recall
    )
    

 
def run_scs(prob, warmstart=None):
    
    # Set up SCS options
    scs_params = PARAMS_SCS_DFLT
    scs_params['verbose']=True
    # Solve
    X, info = prob.cgraph.solve_scs_sparse(setup_kwargs=scs_params, warmstart=warmstart)
    # Process solution
    inliers, er, x_opt = prob.cgraph.process_sdp_var(X)
    precision, recall = prob.get_prec_recall(inliers)
    
    output = dict(
        solver="scs-sparse", 
        size = prob.cgraph.size,
        inliers=inliers,
        er=er,
        t_setup=info['time_setup'],
        t_solve=info['time_solve'],
        t_total = info['time_setup'] + info['time_solve'],
        precision=precision,
        recall=recall
    )
    
    return output

def run_scs_clipper(prob):
    
    # Solve
    inliers, time = prob.solve_clipper()
    precision, recall = prob.get_prec_recall(inliers)
    
    output = dict(
        solver="scs-clipper",
        size = prob.cgraph.size,
        inliers=inliers,
        er=None,
        t_setup=None,
        t_solve=None,
        t_total = time,
        precision=precision,
        recall=recall
    )
    
    return output
 
def compare_sdps(mults = [2,3,4,5,6,7]):
    
    results = []
    for mult in mults:
        m = int(100*mult)
        n1 = int(100*mult)
        n2o = int(10*mult)
        prob = BunnyProb(m=m, n1=n1, n2o=n2o)
        print(f"Running with size {m}")
        # Run Solvers
        print("Sparse SDP...")
        results.append(run_scs(prob=prob))
        # Non-sparse solver
        print("CLIPPER SDP...")
        results.append(run_scs_clipper(prob=prob))

    df = pd.DataFrame(results)
    df.to_pickle('examples/results/sdp_comparison.pkl')
    
def compare_sdps_pp(filename='examples/results/sdp_comparison.pkl'):
    df = pd.read_pickle(filename)
    df.loc[df['solver']=='scs-sparse','size'] = np.array([200, 300,400,500,600,700])
    
    f, ax = plt.subplots(figsize=(7, 7))
    ax.set(xscale="log", yscale="log")    
    sns.lineplot(data=df, ax=ax, hue='solver',x='size', y='t_total')
    plt.show()
 
if __name__ == "__main__":
    # run_scs(mult=5)
    # run_scs(mult=5, warmstart="max-density")
    # test_warmstart_scs(mult=5)
    
    # study_graph_decomp()
    
    # Comparison of SDP runtime
    # compare_sdps()
    compare_sdps_pp()
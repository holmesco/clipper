import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation
import networkx as nx
import matplotlib.pyplot as plt
from time import time
from src.clipper.clipper import ConsistencyGraphProb, generate_bunny_dataset, get_affinity_from_points


class BunnyProb():
    def __init__(self, m = 100, n1 = 100, n2o = 10, outrat = 0.9, sigma = 0.01):
        # Set up common variables for tests
        pcfile = 'examples/data/bun10k.ply'
        T_21 = np.eye(4)
        T_21[0:3, 0:3] = Rotation.random().as_matrix()
        T_21[0:3, 3] = np.random.uniform(low=-5, high=5, size=(3,))
        # Generate dataset
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
        self.cgraph = ConsistencyGraphProb(affinity=self.affinity)
        
    def solve_clipper(self):
        """Solve SDP using CLIPPER formulation"""
        t0 = time()
        self.clipper.solve_as_msrc_sdr()
        t1 = time()
        
        soln = self.clipper.get_solution()
        inliers = np.zeros(self.prob.size)
        inliers[soln.nodes] = 1
        
        return inliers, t1-t0
    
    def solve_fusion_sparse(self, verbose=False):
        X, info = self.cgraph.solve_fusion_sparse(verbose=verbose)
        inliers, er, x_opt = self.cgraph.process_sdp_var(X)
        return inliers, info['time_setup'], info['time_solve']
    
if __name__ == "__main__":
    mult = 2
    prob = BunnyProb(m=100*mult, n1=100*mult, n2o=10*mult)
    
    inliers, t_setup, t_solve = prob.solve_fusion_sparse(verbose=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 17:42:57 2024

@author: saurabh

# explicit FDM

"""

# import libraries
from optionclass import EuropeanCall, EuropeanPut
import numpy as np
from matrix_gen import tridiagonal_matrix_gen, vector_d_gen
from bs_heat_eqn_conversions import *
from thomas import thomas_algo
from SOR import SOR_algo

# Finite Difference Method

class FDM :
    
    def __init__(self, M, N, option, x_min, x_max) :
        
        # member variables
        self.M = M #number of time discretization # tau index in grid : 0 to M
        self.N = N #number of space discretization # x index in grid : 0 to N
        self.x_min = x_min # lower limit for grid space
        self.x_max = x_max # upper limit for grid space
        self.option = option 
        self.tau_max = 0.5 * option.T * option.vol**2 # tau at maturity
        self.dt = self.tau_max / self.M # grid step size
        self.dx = (x_max - x_min) / N
        self.lmbda = self.dt / self.dx ** 2 
        
        self.x_arr = np.linspace(self.x_min, self.x_max, self.N+1)
        self.tau_arr = np.linspace(0, self.tau_max, self.M+1)
        
            
        
        
        
 
            
        
        
       
        
        
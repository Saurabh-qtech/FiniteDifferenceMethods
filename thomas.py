#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:19:35 2024

@author: saurabh
"""

import numpy as np

class thomas_algo :
    
    def __init__(self, A, b) :
        '''
        constructor of class: thomas_algo
        Parameters
        ----------
        A : np.array
            Tridiagonal matrix (Coefficient matix)
        
        b : np.array
            1-D array (constant vector)
        Returns
        -------
        None.

        '''
        self.A = A  # tri-diagonal matrix/ coefficients
        self.dimension = A.shape[0] # number of rows in tri-diagonal matrix
        self.b = b # constant vector
        self.x = np.array([None] * self.dimension) # solution vector
        
    
    def forward_loop (self) :
        '''
        computes step1 (forward loop) of thomas algorithm

        Returns
        -------
        (alpha_hat, b_hat) : tuple (np.array, np.array)
            two 1-D array 

        '''
        
        # transformation after setting sub-diagonal = 0
        alpha_hat = np.array([None] * self.dimension)
        b_hat = np.array([None] * self.dimension)
        
        alpha_hat[0] = self.A[0][0]
        b_hat[0] = self.b[0]
        
        # run forward loop
        for i in range(1, self.dimension) :
            div = self.A[i][i-1]/alpha_hat[i-1]
            alpha_hat[i] = self.A[i][i] - self.A[i-1][i] * div
            b_hat[i] = self.b[i] - b_hat[i-1] * div
            
        return alpha_hat, b_hat
    
    
    def solve (self) :
        '''
        computes step2 (backward loop) of thomas algorithm

        Returns
        -------
        x : np.array
            1-D array (variable/solution vector)

        '''
        
        #source transformation from forward_loop
        alpha_hat, b_hat = self.forward_loop()
        
        #solve for solution vector x
        self.x[self.dimension - 1] = b_hat[self.dimension - 1] / alpha_hat[self.dimension - 1]
        
        for i in range(self.dimension - 2, -1, -1) :
            self.x[i] = (b_hat[i] - self.A[i][i+1]*self.x[i+1])/alpha_hat[i]
         
        # return solution vector   
        return self.x
        
    
            
        
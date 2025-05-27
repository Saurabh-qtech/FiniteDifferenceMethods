#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:11:43 2024

@author: saurabh
"""

import numpy as np


class SOR_algo :
    
    def __init__(self, A, b, omega, tol=1e-6, max_iter=10000) :
        '''
        constructor of class SOR 

        Parameters
        ----------
        A : np.array
            NxN Coefficient matrix.
        b : np.array
            1-D array ie constant vector.
        omega : float
            relaxation factor (0 < omega < 2).
        tol : float, optional
            convergence tolerance level. The default is 1e-6.
        max_iter : int, optional
            maximum number of iterations. The default is 10000.

        Returns
        -------
        x : np.array
            solution vector

        '''
        
        self.A = A  # tri-diagonal matrix/ coefficients matrix
        self.b = b # constant vector
        self.N = len(b) # number of equations
        self.omega_check(omega) # check is 0 < omega < 2
        self.omega = omega #relaxation factor
        self.tol = tol # convergence tolerance
        self.max_iter = max_iter # maximum number of iterations
        self.x = np.array([None] * self.N) # solution vector
        
    def omega_check(self,omega):
        '''
        Error handling for omega passsed as input at instatiation of SOR object

        Parameters
        ----------
        omega : float
            relaxation factor (0 < omega < 2).

        Raises
        ------
        ValueError
            Omega must lie in (0,2) interval.

        Returns
        -------
        None.

        '''
        if omega <= 0 or omega >= 2:
            raise ValueError("Omega must be between 0 and 2 for SOR to converge.")
        
    def solve(self):
        '''
        runs SOR solver

        Returns
        -------
        np.array
            1-D array with solution of system of linear equation.

        '''
        
        x = np.zeros(self.N)  # intial guess
        
        for it in range(self.max_iter):
            
            x_new = np.copy(x)
            
            # next iteration x_new
            for i in range(self.N) :
                
                # lower sum ie (L.x(k+1))
                lsum = np.dot(self.A[i, :i], x_new[:i])
                # upper sum ie (U.x(k))
                usum = np.dot(self.A[i, i+1:], x[i+1:])
                
                x_new[i] = (1 - self.omega) * x[i] + (self.omega / self.A[i,i]) * (self.b[i] - lsum - usum)
            
            
            #convergence check
            if np.linalg.norm(x_new - x, ord = np.inf) < self.tol:
                #print(f'Converged after {it} iterations with omega = {self.omega}')
                self.x = x_new
                return self.x
                
            x = x_new
            
        print('SOR method did not converge within the maximum number of iterations with omega = {self.omega}')
        self.x = x
        return self.x
        



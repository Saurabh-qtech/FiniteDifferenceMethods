#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 00:44:06 2024

@author: saurabh
"""

import numpy as np


def solve_psor(A, b, g, omega, tol=1e-6, max_iter=10000):
    """
    Solves the LCP using Projected Successive Over-Relaxation (PSOR).

    Parameters:
        A (ndarray): Coefficient matrix (n x n).
        b (ndarray): Right-hand side vector (n-dimensional).
        g (ndarray): Lower bound for w (n-dimensional).
        w0 (ndarray): Initial guess for w (n-dimensional).
        omega (float): Relaxation parameter (0 < omega < 2).
        tol (float): Convergence tolerance (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 10000).

    Returns:
        w (ndarray): Solution vector.
        n_iter (int): Number of iterations performed.
        success (bool): Whether the method converged.
    
    n = A.shape[0]
    w = w0.copy()

    for k in range(max_iter):
        w_old = w.copy()
        
        for i in range(n):
            # Compute residual
            r = np.dot(A[i, :], w) - b[i]
            
            # Update w[i] using the PSOR formula
            w[i] = max(g[i], w[i] + omega * (b[i] - r) / A[i, i])
        
        # Check convergence: norm of the difference between iterations
        if np.linalg.norm(w - w_old, ord=np.inf) < tol:
            return w, k + 1, True  # Converged
    
    return w, max_iter, False  # Did not converge
    """
    
    p = A.shape[0]  # option.N-1
    w_init = np.zeros(p)  # intial guess
    x_init = w_init - g
    b_hat = b - np.dot(A,g)
    
    for it in range(max_iter):
        
        x_new = np.copy(x_init)
        
        r = np.zeros(p)
        
        for i in range(p) :
            
            # lower sum ie (L.x(k+1))
            lsum = np.dot(A[i, :i], x_new[:i])
            # upper sum ie (U.x(k))
            usum = np.dot(A[i, i+1:], x_init[i+1:])
            
            r[i] = b_hat[i] - lsum -A[i,i]*x_init[i] - usum
            
            x_new[i] = np.maximum(0, x_init[i] + omega * r[i]/A[i,i])
            
        #convergence check
        if np.linalg.norm(x_new - x_init, ord = np.inf) < tol:
            #print(f'Converged after {it} iterations with omega = {omega}')
            x_init = x_new
            return x_init
        
        x_init = x_new
        
    print('Projected SOR method did not converge within the maximum number of iterations with omega = {omega}')
    return x_init
            
            
            
            
            
            
            
            
    
    

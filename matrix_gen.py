#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:28:29 2024

@author: saurabh
"""

# import libraries
import numpy as np
from scipy.sparse import spdiags

# generate an NxN tridiagonal matrix
def tridiagonal_matrix_gen (n, diag, sub_diag, super_diag):
    '''
    generates a tridiagonal matrix with main diagonal  and super,sub-diagonals 

    Parameters
    ----------
    n : integer
        size of square matrix. 
    
    diag : float
        main diagonal with value = diag
        
    sub_diag : float
        sub diagonal with value = sub_diag

    super_diag : float
        super diagonal with value = super_diag

    Returns
    -------
    A : np.array
        return triadiagonal matrix of size n x n.

    '''
    
    maindiag = diag * np.ones(n)
    subdiag =  sub_diag * np.ones(n)
    superdiag =  super_diag * np.ones(n)
    data = np.array([maindiag, subdiag, superdiag])
    triset = np.array([0, -1, 1])
    
    A = spdiags(data, triset ,n, n).toarray()
    
    return A

# generate vector with first and last row as input and middle rows = 0

def vector_d_gen (n, first, last) :
    
    d = np.zeros(n)
    d[0] = first
    d[n-1] = last
    
    return d


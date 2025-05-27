#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:25:52 2024

@author: saurabh

normal_cdf.py : define function to calculate CDF of standard normal distribution

"""

# import libraries
import numpy as np


# normal pdf of a random variable

def f (x) :
    '''
    

    Parameters
    ----------
    x : float
        value of a random variable following standard normal distribution.

    Returns
    -------
    pdf : float
        output from pdf of a standard normal distribution.

    '''
    
    pdf = (1/np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)      # standard normal distribution formula
    
    return pdf



# approxiamate normal_cdf using formula in HW

def HW_cdfFormula (x) :
    '''
    Calculate CDF of a standard normal distribution
    
    refer HW documentation / Appendix D2 (page 307 - 309 for further information on function)

    Parameters
    ----------
    x : float
        value of a random variable following standard normal distribution.

    Returns
    -------
    cdf : float
        cdf value of a random variable.

    '''
    
    # constants
    a1 = 0.319381530
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    
    if x >= 0 :
        z = 1 / (1 + 0.2316419 * x)
        cdf = 1 - f(x) * z * ((((a5*z + a4)*z + a3)*z + a2)*z + a1)
        
    
    else :
        z = 1 / (1 - 0.2316419 * x)
        cdf = f(-x) * z * ((((a5*z + a4)*z + a3)*z + a2)*z + a1)
        
    return cdf
    
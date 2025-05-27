#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:51:35 2024

@author: saurabh

functions for converting from Black-Scholes PDE to Heat Eq and vice versa
"""

# import libraries
#from optionclass import Option
import numpy as np

# heat equation - BlackScholes equation conversion
# Y to V

def calcVfromY (option, x, tau, Y):
    
    
    alpha = -0.5 * (option.q_y - 1)
    beta = -0.25 * (option.q_y - 1)**2 - option.q
    
    v = (option.K) * np.exp(alpha * x + beta * tau) * Y
    
    return v


# BlackScholes - heat equation conversion
# V to Y

def calcYfromV (option, S, t, v):
    
    x = np.log(S / option.K)
    tau = (option.T - t) * option.vol**2 * 0.5
    
    
    alpha = -0.5 * (option.q_y - 1)
    beta = -0.25 * (option.q_y - 1)**2 - option.q
    
    Y = np.exp(-alpha * x - beta * tau) * v / (option.K)
    
    return Y
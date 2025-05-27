#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:35:48 2024

@author: saurabh

# American option FDM

"""

# import libraries

from fdmclass import FDM
from matrix_gen import *
import numpy as np
from optionclass import AmericanCall, AmericanPut
from SOR_proj import *
from bs_heat_eqn_conversions import *


class AmericanOptionFDM (FDM) :
    
    def __init__(self, M, N, option, x_min, x_max, fdmType) :
        
        super().__init__(M, N, option, x_min, x_max)
        self.fdmType = fdmType # 0: explicit ; 1: implicit ; 2 : C-N
        
        if fdmType == 0 :
            self.theta = 0
        elif fdmType == 1 :
            self.theta = 1
        else :
            self.theta = 0.5
            
        self.A = tridiagonal_matrix_gen(self.N - 1, 1+2*self.lmbda*self.theta, -self.lmbda*self.theta, -self.lmbda*self.theta)
        self.g = self.g_func()
        self.b = self.b_func()
    
    def g_func(self) :
        
        if isinstance(self.option, AmericanCall) :
            
            q_y = self.option.q_y
            q = self.option.q
            
            f = lambda x, tau : np.exp(0.25*tau*((q_y-1)**2 + 4*q))*np.maximum(np.exp(0.5*x*(q_y+1)) - np.exp(0.5*x*(q_y-1)), 0)
            return f
        
        if isinstance(self.option, AmericanPut) :
            
            q_y = self.option.q_y
            q = self.option.q
            
            f = lambda x, tau : np.exp(0.25*tau*((q_y-1)**2 + 4*q))*np.maximum(np.exp(0.5*x*(q_y-1)) - np.exp(0.5*x*(q_y+1)), 0)
            return f
        
    def b_func (self) :
        
        B = tridiagonal_matrix_gen(self.N - 3, -2, 1, 1)
        alpha = self.lmbda * (1 - self.theta)
        
        f = lambda w , w_1, w_m1: w + alpha*np.dot(B,w) + alpha*vector_d_gen(self.N - 3, w_1, w_m1)
        
        return f
    
    def b_1 (self, w1, w2, g0, g_f) :
        
        alpha = self.lmbda * (1 - self.theta)
        return w1 + alpha*(w2 - 2*w1 + g0) + self.lmbda * self.theta * g_f
    
    def b_m1 (self, w_m1, w_m2, g_m, g_f) :
        
        alpha = self.lmbda * (1 - self.theta)
        return w_m1 + alpha*(w_m2 - 2*w_m1 + g_m) + self.lmbda * self.theta * g_f
    
    def price(self) :
        
        w_curr = self.g(self.x_arr[1:-1], 0)  #w_1...w_n-1
        
        for j in range(0, self.M) :
            
            tau_curr = j*self.dt
            tau_next = (j+1)*self.dt
            g_vector =  self.g(self.x_arr[1:-1], tau_next)  # g_1 ,...., g_n1
            
            b_vector = np.zeros(self.N-1) #b1,.....,b_n1
            g0_v = self.g(self.x_arr[0], tau_curr) # g0
            g0_f = self.g(self.x_arr[0], tau_next) # g0_np1
            
            b_vector[0] = self.b_1(w_curr[0], w_curr[1],g0_v , g0_f) #b1
            b_vector[1:-1] = self.b(w_curr[1:-1], w_curr[0], w_curr[-1]) # b2,....,b_n2
            
            gm_v = self.g(self.x_arr[-1], tau_curr) # gm1
            gm_f = self.g(self.x_arr[-1], tau_next) # gm_np1
            b_vector[-1] = self.b_m1(w_curr[-1], w_curr[-2], gm_v, gm_f)
            
            x = solve_psor(self.A, b_vector, g_vector , 1.1)
            
            w_curr = x + g_vector
            
            
        S_index = int(0.5*self.N - 1)
        
        y_sol = w_curr[S_index]
        
        p = calcVfromY (self.option, np.log(self.option.S0 / self.option.K), self.tau_max, y_sol)
        
        return p
        
        
        

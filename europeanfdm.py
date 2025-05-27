#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 01:11:49 2024

@author: saurabh
"""

# import libraries
from fdmclass import FDM
from matrix_gen import *
import numpy as np
from optionclass import *
import numpy as np
from matrix_gen import tridiagonal_matrix_gen, vector_d_gen
from bs_heat_eqn_conversions import *
from thomas import thomas_algo
from SOR import SOR_algo


class EuropeanOptionFDM (FDM) :
    
    def __init__(self, M, N, option, x_min, x_max) :
        
        super().__init__(M, N, option, x_min, x_max)
        
        self.r1 = self.boundary_cond_r1() #***
        self.r2 = self.boundary_cond_r2() # ***
        
        self.initial_y = self.initial_cond(self.x_arr)
        
    # boundary condition : at x_min
    def boundary_cond_r1 (self) :   # ***
        
        if isinstance(self.option, EuropeanCall) :
            f = lambda tau : 0
            return f
        if isinstance(self.option, EuropeanPut) :
            f = lambda tau : np.exp(0.5*(self.option.q_y - 1)*self.x_min + 0.25*(self.option.q_y - 1)**2 * tau)
            return f
        
    # boundary condition : at x_max
    def boundary_cond_r2 (self) :   # ***
        
        if isinstance(self.option, EuropeanCall) :
            f = lambda tau : np.exp(0.5*(self.option.q_y + 1)*self.x_max + tau * 0.25*(self.option.q_y + 1)**2)
            
        if isinstance(self.option, EuropeanPut) :
            f = lambda tau : 0
            
        return f
        
    # boundary condition : at tau = 0
    def initial_cond (self, x) :
        
        if isinstance(self.option, EuropeanCall) :
            #payoff = lambda x : np.maximum(np.exp(0.5*x*(self.option.q_y + 1)) - np.exp(0.5*x*(self.option.q_y - 1)), 0)
            return np.maximum(np.exp(0.5*x*(self.option.q_y + 1)) - np.exp(0.5*x*(self.option.q_y - 1)), 0)
            
            
        if isinstance(self.option, EuropeanPut) :
            #payoff = lambda x : np.maximum(np.exp(0.5*x*(self.option.q_y - 1)) - np.exp(0.5*x*(self.option.q_y + 1)), 0)
            return np.maximum(np.exp(0.5*x*(self.option.q_y - 1)) - np.exp(0.5*x*(self.option.q_y + 1)), 0)
            
        
# Explicit method implementation       
class ExplicitFDM (EuropeanOptionFDM) :
    
    def __init__(self, M, N, option, x_min, x_max) :
        
        # base class constructor
        super().__init__(M, N, option, x_min, x_max)
        self.A = tridiagonal_matrix_gen(self.N-1, 1 - 2*self.lmbda, self.lmbda, self.lmbda)
        
    def price(self) :
        
        y_curr = self.initial_y[1:self.N]
        count = 0
        for j in range(1,self.M+1) :
            
            d = vector_d_gen(self.N - 1, self.r1((j-1)*self.dt), self.r2((j-1)*self.dt))
            y_next = np.dot(self.A , y_curr) + self.lmbda * d
            y_curr = y_next
            count = j
            
        S_index = int(0.5*self.N - 1)
        
        y_sol = y_curr[S_index]
        
        p = calcVfromY (self.option, np.log(self.option.S0 / self.option.K), self.tau_max, y_sol)
        
        return p
    

# Implicit method implementation   
           
class ImplicitFDM (EuropeanOptionFDM) :
      
    def __init__(self, M, N, option, x_min, x_max, solverType = 0, relaxFactor = 1.1) :
        
        # base class constructor
        super().__init__(M, N, option, x_min, x_max)
        self.A = tridiagonal_matrix_gen(self.N-1, 1 + 2*self.lmbda, -self.lmbda, -self.lmbda)
        self.solverType = solverType  # 0: Thomas Algo; 1: SOR
        self.omega = relaxFactor # relaxation factor used in sor algorithm
        
    def price(self) :
        
        y_curr = self.initial_y[1:self.N] # time = j-1 / constant vector
        
        for j in range(1, self.M+1) :
            
            d = vector_d_gen(self.N - 1, self.r1((j)*self.dt), self.r2((j)*self.dt))
            
            if self.solverType == 0 :
                
                thomas = thomas_algo (self.A, y_curr + self.lmbda * d) # initilize instance of thomas_algo
                y_next = thomas.solve() # call member function to generate solution vector
                y_curr = y_next
                
            if self.solverType == 1 :
                
                sor = SOR_algo (self.A, y_curr + self.lmbda * d, self.omega) # initilize instance of thomas_algo
                y_next = sor.solve() # call member function to generate solution vector
                y_curr = y_next
                
        
        S_index = int(0.5*self.N - 1)
        
        y_sol = y_curr[S_index]
        
        p = calcVfromY (self.option, np.log(self.option.S0 / self.option.K), self.tau_max, y_sol)
        
        return p
                
    
# CrankNicolson method implementation   
           
class CrankNicolsonFDM (EuropeanOptionFDM) :
      
    def __init__(self, M, N, option, x_min, x_max, solverType = 0, relaxFactor = 1.1) :
        
        # base class constructor
        super().__init__(M, N, option, x_min, x_max)
        self.A = tridiagonal_matrix_gen(self.N-1, 1 + self.lmbda, -0.5*self.lmbda, -0.5*self.lmbda)
        self.B = tridiagonal_matrix_gen(self.N-1, 1 - self.lmbda, 0.5*self.lmbda, 0.5*self.lmbda)
        
        self.solverType = solverType  # 0: Thomas Algo; 1: SOR
        self.omega = relaxFactor # relaxation factor used in sor algorithm
        
    def price(self) :
        
        y_curr = self.initial_y[1:self.N] # time = j-1 / constant vector
        
        for j in range(1, self.M+1) :
            
            d = vector_d_gen(self.N - 1, self.r1((j-1)*self.dt)+self.r1((j)*self.dt), self.r2((j)*self.dt)+self.r2((j-1)*self.dt))
            
            c = np.dot(self.B , y_curr) + 0.5 * self.lmbda * d
            
            if self.solverType == 0 :
                
                thomas = thomas_algo (self.A, c) # initilize instance of thomas_algo
                y_next = thomas.solve() # call member function to generate solution vector
                y_curr = y_next
                
            if self.solverType == 1 :
                
                sor = SOR_algo (self.A, c, self.omega) # initilize instance of thomas_algo
                y_next = sor.solve() # call member function to generate solution vector
                y_curr = y_next
                
        
        S_index = int(0.5*self.N - 1)
        
        y_sol = y_curr[S_index]
        
        p = calcVfromY (self.option, np.log(self.option.S0 / self.option.K), self.tau_max, y_sol)
        
        return p
       
    
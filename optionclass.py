#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 09:00:13 2024

@author: saurabh

# create classes for different financial instruments

"""

# import libraries
import numpy as np

# Base class : Option
class Option :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # member variables
        self.S0 = S0
        self.y = y
        self.vol = vol
        self.K = K
        self.r = r
        self.T = T
        
        # member variables : heat eqn transformation
        self.q = 2 * self.r / (self.vol ** 2)
        self.q_y = 2 * (self.r - self.y) / (self.vol ** 2)
        
        
# Derived class : OptionType = European
class European (Option) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionType = 'european'
        
# Derived class : OptionType = European
class American (Option) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionType = 'american'
        


# Derived class : OptionSubType = Call        
class EuropeanCall (European) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionSubType = 'call'
        
    # payoff
    def payoff (self,S):
        
        payoff = np.maximum(S - self.K, 0)
        return payoff
        
    


# Derived class : OptionSubType = Put  
class EuropeanPut (European) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionSubType = 'put'
        
    # payoff
    def payoff(self, S):
        
        payoff = np.maximum(self.K - S, 0)
        return payoff
    
# Derived class : OptionSubType = Call        
class AmericanCall (American) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionSubType = 'call'
        
    # payoff
    def payoff (self,S):
        
        payoff = np.maximum(S - self.K, 0)
        return payoff
        
    


# Derived class : OptionSubType = Put  
class AmericanPut (American) :
    
    # constructor
    def __init__(self, S0, y, vol, K, r, T):
        
        # parent class constructor
        super().__init__(S0, y, vol, K, r, T)
        self.optionSubType = 'put'
        
    # payoff
    def payoff(self, S):
        
        payoff = np.maximum(self.K - S, 0)
        return payoff
        
    
        
        
        
        
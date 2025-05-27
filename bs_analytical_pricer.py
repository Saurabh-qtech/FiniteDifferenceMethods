#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:11:29 2024

@author: saurabh

bs_analytical_pricer.py : create a class with member function to calculate price of European calls and puts
using the famous Black Scholes analytical formula. Additional member function that can be added 
to the class are for calculating greeks using analytical formulas.

"""

import numpy as np
from normal_cdf import HW_cdfFormula
from optionclass import *


# Package for analytical price using black scholes model
class AnalyticalPricer:
    
    def price(self, option):
        '''
        Calculate price of an European option passed in the function argument using the analytical formula of Black Scholes

        Parameters
        ----------
        option : base class - EuropeanOption inherited class - EuropeanCall/EuropeanPut
            user defined object of class EuropeanCall/EuropeanPut

        Returns
        -------
        value : float
            value of option passed in the argument.

        '''
        
        # unwrap option parameters
        S = option.S0
        K = option.K
        r = option.r
        t = 0       # current time = 0
        T = option.T
        y = option.y
        sigma = option.vol
        
        # calculate d1 and d2 in BSM formaula
        d1 = (np.log(S/K) + (r -y + 0.5 * sigma**2)*(T-t)) / (sigma * np.sqrt(T-t))
        d2 = d1 - sigma * np.sqrt(T-t)
        
        # BSM formula for call option
        if isinstance(option, EuropeanCall):
            value = S * np.exp(-y*(T-t)) * HW_cdfFormula(d1) - K * np.exp(-r * (T-t)) * HW_cdfFormula(d2)
            
        # BSM formula for put option    
        if isinstance(option, EuropeanPut):
            value = -S * np.exp(-y*(T-t)) * HW_cdfFormula(-d1) + K * np.exp(-r * (T-t)) * HW_cdfFormula(-d2)
            
        return value
        

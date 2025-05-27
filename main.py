#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 08:56:20 2024

@author: saurabh

Course - HW 6 : (Computational Methods in Finance)

Objective : Calculate price of European and American options using Finite Difference Method (Explicit, Implicit and Crank-Nicolson)

Numerical Methods/Algorithms used to solve system of equation:
    1. Thomas Algorithm.
    2. SOR.
    3. Projected SOR


"""

# import libraries
from optionclass import *
from matrix_gen import *
from fdmclass import *
from europeanfdm import *
from bs_analytical_pricer import AnalyticalPricer
from bs_heat_eqn_conversions import *
import pandas as pd
from americanfdm import *
import pandas as pd
from tabulate import tabulate
import numpy as np


def main() :
    
    
    
    # define option parameters
    S0 = 100
    y = 0.025
    vol = 0.75
    K = 100
    r = 0.03
    T = 1
    
    np.set_printoptions(precision=6)

    #----------------------------------
    # Create Options (Eur and American)
    #----------------------------------
    eur_put = EuropeanPut(S0,y,vol,K,r,T)  # Eur put option
    eur_call = EuropeanCall(S0,y,vol,K,r,T) # Eur call option

    am_put = AmericanPut(S0,y,vol,K,r,T)        # american put option
    am_call = AmericanCall(S0,y,vol,K,r,T)      # american call option


    #----------------------------------------
    # Explicit pricer (European and American)
    #----------------------------------------

    # I. European options
    #--------------------
    
    # Call
    Explict_pricer_ec = ExplicitFDM(800, 100, eur_call, -2.5, 2.5)  # Explicit Grid create
    Explicit_price_ec = Explict_pricer_ec.price()                   # call price member function
    
    # Put
    Explict_pricer_ep = ExplicitFDM(800, 100, eur_put, -2.5, 2.5)   # Explicit Grid create
    Explicit_price_ep = Explict_pricer_ep.price()                   # call price member function
    
    # II. American options
    #----------------------
    
    # Call
    am_explicit_call = AmericanOptionFDM(800,100, am_call, -2.5, 2.5,0) # Explicit Grid create
    am_explicit_call_price = am_explicit_call.price()                   # call price member function

    # Put
    am_explicit_put = AmericanOptionFDM(800,100, am_put, -2.5, 2.5,0)  # Explicit Grid create
    am_explicit_put_price = am_explicit_put.price()             # call price member function



    #----------------------------------------
    # Analytic Price using BS formula
    #----------------------------------------
    
    BS_analytic = AnalyticalPricer()    # Create pricer object
    # Call
    Anlaytic_ec = BS_analytic.price(eur_call) # call price member function
    # Put
    Anlaytic_ep = BS_analytic.price(eur_put)    # call price member function


    #----------------------------------------
    # Implicit pricer (European and American)
    #----------------------------------------

    # I. Thomas algo:
    #-----------------

    # I.a. European
    # ---------------
    
    # Call
    Implict_pricer_ec_thomas = ImplicitFDM(800, 100, eur_call, -2.5, 2.5,0) # Implicit Grid Create
    Implicit_price_ec_thomas = Implict_pricer_ec_thomas.price()     # call price member function

    # Put
    Implict_pricer_ep_thomas = ImplicitFDM(800, 100, eur_put, -2.5, 2.5,0)  # Implicit Grid Create
    Implicit_price_ep_thomas = Implict_pricer_ep_thomas.price()         # call price member function
        
    
    # II. SOR algo:
    #---------------

    # II. a. European (Using SOR)
    #----------------------------
    
    # Call
    Implict_pricer_ec_sor = ImplicitFDM(800, 100, eur_call, -2.5, 2.5,1,1.1) # Implicit Grid Create
    Implicit_price_ec_sor = Implict_pricer_ec_sor.price()           # call price member function

    # Put
    Implict_pricer_ep_sor = ImplicitFDM(800, 100, eur_put, -2.5, 2.5,1,1.1)     # Implicit Grid Create
    Implicit_price_ep_sor = Implict_pricer_ep_sor.price()           # call price member function

    # II. b. American (Using Projected SOR)
    #--------------------------------------
    
    # Call
    am_implicit_call = AmericanOptionFDM(800,100, am_call, -2.5, 2.5,1)         # Implicit Grid Create
    am_implicit_call_price = am_implicit_call.price()           # call price member function

    # Put
    am_implicit_put = AmericanOptionFDM(800,100, am_put, -2.5, 2.5,1)       # Implicit Grid Create
    am_implicit_put_price = am_implicit_put.price()                 # call price member function
    
    #----------------------------------------------
    # Crank-Nicolson pricer (European and American)
    #----------------------------------------------

    # I. Thomas algo:
    #-----------------
    
    # Call
    CN_pricer_ec_thomas = CrankNicolsonFDM(800, 100, eur_call, -2.5, 2.5,0)     # C-N Grid Create
    CN_price_ec_thomas = CN_pricer_ec_thomas.price()                    # call price member function

    # Put
    CN_pricer_ep_thomas = CrankNicolsonFDM(800, 100, eur_put, -2.5, 2.5,0)      # C-N Grid Create
    CN_price_ep_thomas = CN_pricer_ep_thomas.price()                    # call price member function
    
    
    # II. SOR algo:
    #---------------

    # II. a. European using (SOR)
    #----------------------------

    # call
    CN_pricer_ec_sor = CrankNicolsonFDM(800, 100, eur_call, -2.5, 2.5,0,1.1)     # C-N Grid Create
    CN_price_ec_sor = CN_pricer_ec_sor.price()                          # call price member function

    # Put
    CN_pricer_ep_sor = CrankNicolsonFDM(800, 100, eur_put, -2.5, 2.5,0,1.1)     # C-N Grid Create
    CN_price_ep_sor = CN_pricer_ep_sor.price()                      # call price member function
    
    

    # II. b. American (Using Projected SOR)
    #--------------------------------------
    
    # Call
    am_cn_call = AmericanOptionFDM(800,100, am_call, -2.5, 2.5,2)       # C-N Grid Create
    am_cn_call_price = am_cn_call.price()                       # call price member function

    # Put
    am_cn_put = AmericanOptionFDM(800,100, am_put, -2.5, 2.5,2)  # C-N Grid Create
    am_cn_put_price = am_cn_put.price()                 # call price member function
    
    
    
    
    # save prices caluculated for european call
    
    summary_eur_call = {
        'Pricing method' : ['Black Scholes Analytic' , 'Explicit', 'Implicit (Thomas)' , 'Implicit (SOR)' ,'CN (Thomas)' ,'CN (SOR)' ],
        'Price' : [Anlaytic_ec, Explicit_price_ec, Implicit_price_ec_thomas, Implicit_price_ec_sor, CN_price_ec_thomas, CN_price_ec_sor]
        }
    eur_call_df = pd.DataFrame(summary_eur_call)
    eur_call_df['abs error'] = abs(eur_call_df['Price'] - Anlaytic_ec)
    
    # save prices caluculated for european put
    summary_eur_put = {
        'Pricing method' : ['Black Scholes Analytic' , 'Explicit', 'Implicit (Thomas)' , 'Implicit (SOR)' ,'CN (Thomas)' ,'CN (SOR)' ],
        'Price' : [Anlaytic_ep, Explicit_price_ep, Implicit_price_ep_thomas, Implicit_price_ep_sor, CN_price_ep_thomas, CN_price_ep_sor]
        
        }
    eur_put_df = pd.DataFrame(summary_eur_put)
    eur_put_df['abs error'] = abs(eur_put_df['Price'] - Anlaytic_ep)
              
     # save prices caluculated for american call    

    summary_am_call = {
        'Pricing method' : ['Explicit', 'Implicit (Projected SOR)' ,'CN (Projected SOR)' ],
        'Price' : [am_explicit_call_price, am_implicit_call_price, am_cn_call_price]
        }
    am_call_df = pd.DataFrame(summary_am_call)
    am_call_df['abs error'] = abs(am_call_df['Price'] - Anlaytic_ec)    
    
    #save prices caluculated for american put
    
    summary_am_put = {
        'Pricing method' : ['Explicit', 'Implicit (Projected SOR)' ,'CN (Projected SOR)' ],
        'Price' : [am_explicit_put_price, am_implicit_put_price, am_cn_put_price]
        }
    am_put_df = pd.DataFrame(summary_am_put)
    am_put_df['abs error'] = abs(am_put_df['Price'] - Anlaytic_ep)   
    
    
    # print data
    print("\nPricing American and European Options using FDM")
    print("Parameters:")
    print(f"S0 = ${S0}, K = ${K}, r = {r}, div = {y}, vol = {vol}, T = {T}")
    print("\nEuropean Call Prices:")
    print(tabulate(eur_call_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".6f"))
    print("\nEuropean Put Prices:")
    print(tabulate(eur_put_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".6f"))
    print('\n American Call Prices:')
    print(tabulate(am_call_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".6f"))
    print('\n American Put Prices:')
    print(tabulate(am_put_df, headers='keys', tablefmt='grid', showindex=False, floatfmt=".6f"))
    
main()
    
    


















    

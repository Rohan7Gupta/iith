# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:46:37 2024

@author: lenovo
"""

#loss
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

"""
Source: Electrooptical  Effects  in  Silicon 
        RICHARD  A.  SOREF
"""
m_star_e = 0.26 * 9.10938356e-31  # Effective mass of electron (kg) 
m_star_h = 0.39 * 9.10938356e-31  #Effective mass of hole (kg)

# Constants
length = 2e-6
epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
epsilon_inf = 11.7           # High-frequency permittivity for Si
e = 1.602176634e-19          # Elementary charge (C)
m_star = 0.26 * 9.10938356e-31  # Effective mass of electron (kg)
wavelength = 1550e-9         # Wavelength of incident light (m)
omega = 2 * np.pi * 3e8 / wavelength  # Angular frequency of light (rad/s)
V_bi = 0.7                   # Built-in potential (V)
gamma = 1e15                 # Collision frequency (1/s)
epsilon_s = epsilon_inf * epsilon_0  # Permittivity of silicon
k = 1.38064852e-23
t = 298 #kelvin
k_0 = 2*np.pi / wavelength
n_i = 1.5e10
# Function to calculate built-in voltage
def calculate_builtin_voltage(N_A, N_D):
    V_bi = (k * t / e) * np.log((N_A * N_D) / (n_i*1e6)**2)
    return V_bi

# Function to calculate depletion width
def depletion_width(V_R, N_A, N_D):
    V_bi = calculate_builtin_voltage(N_A, N_D)
    d=(2 * epsilon_s * (V_bi + abs(V_R)) * (N_A + N_D) / (e * N_A * N_D))
    return np.sqrt(d)

def x_n(V_R, N_A, N_D):
    return depletion_width(V_R, N_A, N_D) * N_A / (N_A + N_D)

def x_p(V_R, N_A, N_D):
    return -depletion_width(V_R, N_A, N_D) * N_D / (N_A + N_D)

"""
 Source: CARRIER CONCENTRATIONS AND DRIFT CURRENTS IN THE DEPLETION REGION OF A p n JUNCTION
         D. K. MAK
"""
def p_x_maj(x, V_R, N_A, N_D): #majority p carrier desity -xp<x<0
    a_p = (e ** 2) * N_A / (2 * epsilon_s * k * t)
    x_p_val = -x_p(V_R, N_A, N_D)
    p = N_A * np.exp(-a_p * (x + x_p_val) ** 2)
    #print("p\t",a_p,N_A,p)
    return p 

def n_x_maj(x, V_R, N_A, N_D): #majority n carrier density 0<x<xn
    a_n = (e ** 2) * N_D / (2 * epsilon_s * k * t)
    x_n_val = x_n(V_R, N_A, N_D)
    n = N_D * np.exp(-a_n * (x - x_n_val) ** 2)
    #print("n\t",a_n,N_D,n)
    return n 

def carrier_p_theory(V_R, N_A, N_D): #input /m3

    xp = x_p(V_R, N_A, N_D)
    l = length/2  # PN junction at centre of waveguide
    if abs(xp) > l:
        carrier_p, _ = quad(p_x_maj, -l, 0, args=(V_R, N_A, N_D))
    else:
        carrier_p, _ = quad(p_x_maj, xp, 0, args=(V_R, N_A, N_D)) + (l+xp)*N_A
    return carrier_p / (length * 1e6  * 0.5) #1e6 multiplied to convert /m3 to /cc

def carrier_n_theory(V_R,N_A,N_D): #input /m3

    xn = x_n(V_R, N_A, N_D)
    l = length/2  # PN junction at centre of waveguide
    if xn > l:
        carrier_n, _ = quad(n_x_maj, 0, l, args=(V_R, N_A, N_D))
    else:
        carrier_n, _ = quad(n_x_maj, 0, xn, args=(V_R, N_A, N_D)) + (l-xn)*N_D
    return carrier_n / (length * 1e6 * 0.5) #1e6 multiplied to convert /m3 to /cc

"""
using kramer - kronig model approximation (previosly verified against the plasma dispersion model)
author    = {Reed, G. T. and Mashanovich, G. and Gardes, F. Y. and Thomson, D. J.}
journal   = {Nature Photonics}
title     = {Silicon optical modulators}
year      = {2010}
"""

def n_effective(V_R, N_A, N_D):
    #return -(5.4e-22 * pow(carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6),1.011) + 1.53e-18 * pow(carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6),0.838))
    return -(8.8e-22 * carrier_n_theory((V_R),N_A*1e6,N_D*1e6) + 8.5e-18 * pow(carrier_p_theory((V_R),N_A*1e6,N_D*1e6),0.8))



def del_phi_eff(V_R, N_A, N_D, LENGTH): #Theory = 1, synopsis = 0, lorentz = 2
    return ((k_0 *abs( n_effective(V_R, N_A, N_D) - n_effective(0, N_A, N_D)) *1e-3 )) % (2*np.pi) #degree phase




def generate_logspace_sweep(start_exp=17, end_exp=21, num_values=500):
    return np.logspace(start_exp, end_exp, num=num_values)

# Generating the sweep values
N_values = generate_logspace_sweep() #Doping => 1e17 to 1e21 /cm^3 

# Generate V_R values
V_R_values = np.arange(0, 10, 0.1)

# Length of the phase shifter in meters
LENGTH = 4e-3

# Create the lookup table (LUT)
lut = []

for V_R in V_R_values:
    row = [V_R]
    for N in N_values:
        N_A = N_D = N  # Assuming N_A = N_D
        phi_eff = del_phi_eff(V_R, N_A, N_D, LENGTH)
        row.append(phi_eff)
    lut.append(row)

# Convert to numpy array for easier handling
lut = np.array(lut)

# Save the LUT to a CSV file
np.savetxt('lut.csv', lut, delimiter=',', header=','.join(['V_R'] + [f'N={N:.10e}' for N in N_values]), comments='')

# Example print statement to verify
print("LUT saved to lut.csv")


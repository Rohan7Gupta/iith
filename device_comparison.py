# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:32:44 2024

@author: Rohan
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import curve_fit

#import math


epsilon_0 = 8.854187817e-12  # Permittivity of free space (F/m)
epsilon_inf = 11.7           # High-frequency permittivity for Si
e = 1.602176634e-19          # Elementary charge (C)

"""
Source: Optical study of undoped, B or P-doped polysilicon 
        Y. Laghla , E. Scheid
"""
# =============================================================================
# m_star_e = 0.26 * 9.10938356e-31  # Effective mass of electron (kg) 
# m_star_h = 0.39 * 9.10938356e-31  #Effective mass of hole (kg)
# 
# =============================================================================

"""
Source: Electrooptical  Effects  in  Silicon 
        RICHARD  A.  SOREF
"""
m_star_e = 0.26 * 9.10938356e-31  # Effective mass of electron (kg) 
m_star_h = 0.39 * 9.10938356e-31  #Effective mass of hole (kg)

c = 299792458 # m/s
wavelength = 1550e-9         # Wavelength of incident light (m)
omega = 2 * np.pi * c / wavelength  # Angular frequency of light (rad/s)
gamma = 1e15                 # Collision frequency (1/s)
epsilon_s = epsilon_inf * epsilon_0  # Permittivity of silicon
k = 1.38064852e-23           # boltzmann constant [J/K]
t = 298                      #kelvin
k_0 = 2*np.pi / wavelength
LENGTH = 4e-3                # Lenght of phase shifter (m)
length = 0.5e-6
n_i = 1.5e10  # Intrinsic carrier concentration for silicon (cm^-3)

"""
Source: Dielectric functions and optical parameters of Si, Ge, GaP, GaAs, GaSb,
        InP, InAs, and InSb from 1.5 to 6.0 eV
        D. E. Aspnes and A. A. Studna
"""
alpha_0 = 0.78 #/cm
n = 3.7 #what is refractive  index of unperturbed  c-Si


# Function to calculate built-in voltage
def calculate_builtin_voltage(N_A, N_D):
    V_bi = (k * t / e) * np.log((N_A * N_D) / (n_i*1e6)**2)
    return V_bi

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


def format_to_4_digits(number):
    # Format the number to a 4-digit string with leading zeros
    formatted_number = f"{number:04}"
    return formatted_number 

"""
Source : Electron and Hole Mobilities in Silicon as a Function of Concentration and Temperature
         N.D. Arora; J.R. Hauser; D.J. Roulston
"""
def calculate_mobilities(N_A, N_D, T=300):

    # Electron mobility parameters for silicon at 300K
    mu_n0 = 65      # cm^2/V·s
    mu_n1 = 1414    # cm^2/V·s
    N_ref_n = 9.68e16  # cm^-3
    gamma_n = 0.68

    # Calculate electron mobility
    mu_n = mu_n0 + (mu_n1 - mu_n0) / (1 + (N_D / N_ref_n) ** gamma_n)


    # Hole mobility parameters for silicon at 300K
    mu_p0 = 48      # cm^2/V·s
    mu_p1 = 470.5   # cm^2/V·s
    N_ref_p = 2.35e17  # cm^-3
    gamma_p = 0.76

    # Calculate hole mobility
    mu_p = mu_p0 + (mu_p1 - mu_p0) / (1 + (N_A / N_ref_p) ** gamma_p)
    return mu_n, mu_p
# read synopsis data
def read_data(N_A,N_D):
    # Load the CSV file
    global length
    """
    Special case as data manually formatted
    """
    if (N_A == 1e17 and N_D == 1e17) :
        file_path = r"C:\Users\lenovo\Documents\spyder\np 1e17 1e17\hDensity vs length_meenakshi csv.csv"
        dfh = pd.read_csv(file_path)
        file_path_e = r"C:\Users\lenovo\Documents\spyder\np 1e17 1e17\eDensity vs length_meenakshi csv.csv"
        dfe = pd.read_csv(file_path_e)
        h_data_by_voltage = {}
        e_data_by_voltage = {}
        voltages =  np.arange(0,-10.1,-0.5)
        #print(dfh.columns)
        # Separate lengths and hDensities for each voltage
        for voltage in voltages:
            format_voltage = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
            #print(format_voltage)
            length_col = f"Length at {format_voltage}V"
            hDensity_col = f"hDensity at {format_voltage}V"
            #print(length_col, hDensity_col)
            if length_col in dfh.columns and hDensity_col in dfh.columns:
                h_data_by_voltage[(format_voltage)] = pd.DataFrame({
                    'Length': dfh[length_col],
                    'hDensity': dfh[hDensity_col]
                })
        # =============================================================================
        #             else:
        #                 print('else',length_col,hDensity_col)    
        #         print(dfh.columns)
        # =============================================================================
        # Separate lengths and eDensities for each voltage
        for voltage in voltages:
            format_voltage = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
            #print(format_voltage)
            length_col = f"Length at {format_voltage}V"
            eDensity_col = f"eDensity at {format_voltage}V"
            #print(length_col, eDensity_col)
            if length_col in dfe.columns and eDensity_col in dfe.columns:
                e_data_by_voltage[(format_voltage)] = pd.DataFrame({
                    'Length': dfe[length_col],
                    'eDensity': dfe[eDensity_col]
                })
            # =============================================================================
            #             else:
            #                 print('else',length_col,hDensity_col)
            # =============================================================================
    else:
        """
        General case for most situation
        """     
        # =============================================================================
        #         if(N_A == 5e19 and N_D == 5e19): pn 1um
        #             file_id = 30
        #             file_path = r"C:\Users\lenovo\Documents\spyder\np 5e19 5e19\hDensity at 5e17 csv.csv"
        #             file_path_e = r"C:\Users\lenovo\Documents\spyder\np 5e19 5e19\eDensity at 5e17 csv.csv"
        #             length = 1e-6 
        # =============================================================================        
        #print(N_A, N_D)
        if(N_A == 5e19 and N_D == 5e19): #pn 0.5um
            file_id = 38
            file_path = r"C:\Users\lenovo\Documents\spyder\data_pn_5e19_5e19_500nm\hDensity p 5e19 n 5e19 csv.csv"
            file_path_e = r"C:\Users\lenovo\Documents\spyder\data_pn_5e19_5e19_500nm\eDensity p 5e19 n 5e19 csv.csv"
        if(N_A == 9e19 and N_D == 1e19): #p+n 0.5um
            file_id = 39
            file_path_e = r"C:\Users\lenovo\Documents\spyder\data_p+n_9e19_1e19_500nm\eDensity n1e19 p 9e19 csv.csv"
            file_path = r"C:\Users\lenovo\Documents\spyder\data_p+n_9e19_1e19_500nm\hDensity n1e19 p9e19 csv.csv"
            length = 1e-6    
        if(N_A ==1e19 and N_D == 9e19): #n+p 1um
            file_id = 33
            file_path = r"C:\Users\lenovo\Documents\spyder\n+p 1e19 9e19\hDensity for n+p csv.csv"
            file_path_e = r"C:\Users\lenovo\Documents\spyder\n+p 1e19 9e19\eDensity for n+p csv.csv"
            length = 1e-6
        if(N_A ==9.95e19 and N_D == 5e17): #p++n 1um
            file_id = 31
            file_path = r"C:\Users\lenovo\Documents\spyder\p+n 9.95e19 5e17\hDensity at p9.95e19 csv.csv"
            file_path_e = r"C:\Users\lenovo\Documents\spyder\p+n 9.95e19 5e17\eDensity at p9.95e19 csv.csv"
            length = 1e-6   
        
        #print(file_path, file_path_e,file_id)
        
        dfh = pd.read_csv(file_path)
        dfe = pd.read_csv(file_path_e)        
        h_data_by_voltage = {}
        e_data_by_voltage = {}
        voltages =  np.arange(0,-5.1,-0.5)       
        #print(dfh.columns)
        # Separate lengths and hDensities for each voltage
        for voltage in voltages:
            format_voltage = format_to_4_digits(int(abs(voltage)/0.5))
            length_col = f"hDensity(C1(PnJunction{file_id}__{format_voltage}_des)) X"
            hDensity_col = f"hDensity(C1(PnJunction{file_id}__{format_voltage}_des)) Y"
            #print(length_col, hDensity_col)
            format_voltage2 = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
            if length_col in dfh.columns and hDensity_col in dfh.columns:
                h_data_by_voltage[(format_voltage2)] = pd.DataFrame({
                    'Length': dfh[length_col],
                    'hDensity': dfh[hDensity_col]
                })
            #else:
                #print(length_col,hDensity_col)
            
        #print(dfh.columns)
        # Separate lengths and eDensities for each voltage
        for voltage in voltages:
            format_voltage = format_to_4_digits(int(abs(voltage)/0.5))
            #print(format_voltage)
            length_col = f"eDensity(C1(PnJunction{file_id}__{format_voltage}_des)) X"
            eDensity_col = f"eDensity(C1(PnJunction{file_id}__{format_voltage}_des)) Y"
            format_voltage2 = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
            if length_col in dfe.columns and eDensity_col in dfe.columns:
                e_data_by_voltage[(format_voltage2)] = pd.DataFrame({
                    'Length': dfe[length_col],
                    'eDensity': dfe[eDensity_col]
                })
            #else:
                #print(length_col,hDensity_col)     
    return h_data_by_voltage,e_data_by_voltage

def plot_charge_profile(N_A,N_D):
    
    data_dict_h,data_dict_e = read_data(N_A, N_D)
    
    plt.figure(figsize = (10,6))
    # Plot hDensity vs Length for each voltage in data_dict_h
    for key, dfh in data_dict_h.items():
        hLength = dfh['Length']
        hDensity = dfh['hDensity']
        plt.plot(hLength, hDensity, label=f'p {key}V')

    for key, dfe in data_dict_e.items():
        eLength = dfe['Length']
        eDensity = dfe['eDensity']
        plt.plot(eLength, eDensity, linestyle='--', label=f'n {key}V')

    plt.yscale('log')
    plt.xlabel(r'Length ($\mu$m)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel('Density', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(f'Charge Profile $N_A = {N_A}$ $N_D = {N_D}$', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.legend()
    plt.show()


#Lorentz model 
def plasma_frequency_e(V_R, N_A, N_D): #drude model
    c_n = carrier_n_theory(V_R, N_A*1e6, N_D*1e6)  # /cc
    return np.sqrt(abs(c_n*1e6 * e**2 / (epsilon_0  * m_star_e))) 

def plasma_frequency_h(V_R, N_A, N_D): #drude model
    c_p = carrier_p_theory(V_R, N_A*1e6, N_D*1e6)  
    return np.sqrt(abs(c_p*1e6 * e**2 / (epsilon_0 * m_star_h))) 

# =============================================================================
# def plasma_frequency(V_R,N_A,N_D):
#     return np.sqrt(plasma_frequency_e(V_R, N_A, N_D)**2 + plasma_frequency_h(V_R, N_A, N_D)**2)
# 
# =============================================================================
def permittivity(V_R, N_A, N_D):
    omega_p = (N_D*plasma_frequency_e(V_R, N_A, N_D) + N_A*plasma_frequency_h(V_R, N_A, N_D))/(N_A + N_D)
    """
    taking weighted average of e & h plasma frequency
    """
    return epsilon_inf - omega_p**2 / (omega**2 + 1j * gamma * omega)

# =============================================================================
# def permittivity(V_R,N_A,N_D):
#     omega_p_h = plasma_frequency_h(V_R, N_A, N_D)
#     omega_p_e = plasma_frequency_e(V_R, N_A, N_D)
#     return epsilon_inf - ( omega_p_h**2 / (omega**2 + 1j * gamma * omega) + omega_p_e**2 / (omega**2 + 1j * gamma * omega) )
# 
# =============================================================================

# Function to calculate total hDensity across total length for a particular voltage
def carrier_p(voltage,N_A,N_D): #/cc
    
    data_dict_h,_ = read_data(N_A, N_D)
    
    dfh = data_dict_h[str(voltage)]
    hLength = dfh['Length']
    hDensity = dfh['hDensity']

    # Calculate the integral using trapezoidal rule 
    integral_value = trapz(hDensity, hLength)
    return integral_value / (length * 1e6 * 0.5) #length 0 - 0.5 but unit um

def carrier_n(voltage,N_A,N_D): #/cc
    
    _,data_dict_e = read_data(N_A, N_D)
    
    dfe = data_dict_e[str(voltage)]
    eLength = dfe['Length']
    eDensity = dfe['eDensity']

    # Calculate the integral using trapezoidal rule 
    integral_value = trapz(eDensity, eLength)
    return integral_value / (length * 1e6 * 0.5) #length 0 - 0.5 but unit um

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
Source: Silicon optical modulators  
        G.T. Reed, G. Mashanovich, F.Y. Gardes and D. J. Thomson 
        derived for 1.55 um from Electrooptical  Effects  in  Silicon 
                                 RICHARD  A.  SOREF, S
"""
def n_effective(voltage, N_A, N_D,theory):  
    #Theory_Plasma = 1, synopsis_Plasma = 0, lorentz = 2, Theory_kk = 3, synopsis_kk = 4
    V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
    constant = ((e ** 2) * ((wavelength) ** 2))/ (8 * (np.pi**2) * ((c)**2) * (epsilon_0 ) * n)
    if theory == 1:
        #converting /cc to /m3
        return -constant * (carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6)/(m_star_e*1e-6) + carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)/(m_star_h*1e-6))
    elif theory == 0:
        #print(V_R,type(V_R))
        return -constant * ((carrier_n((V_R),N_A,N_D)/(m_star_e*1e-6)) + (carrier_p((V_R),N_A,N_D)/(m_star_h*1e-6)))
    elif theory == 2:
        epsilon = permittivity(float(V_R), N_A, N_D)
        return (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)+ np.real(epsilon))/2)))
    elif theory == 3:
        return -(8.8e-22 * carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6) + 8.5e-18 * pow(carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6),0.8))
    elif theory == 4:
        return -(8.8e-22 * carrier_n(V_R,N_A,N_D) + 8.5e-18 * pow(carrier_p(V_R,N_A,N_D),0.8))

"""
Source: Electrooptical  Effects  in  Silicon 
        RICHARD  A.  SOREF, S
"""

#giving similar result to prev approach

# =============================================================================
# def n_effective(voltage, N_A, N_D,theory):  #Theory = 1, synopsis = 0, lorentz = 2
#     V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
#     constant = ((e ** 2) * ((wavelength) ** 2))/ (8 * (np.pi**2) * ((c)**2) * (epsilon_0 ) * n)
#     if theory == 1:
#         #converting /cc to /m3
#         return -constant * (carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6)/(m_star_e*1e-6) + carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)/(m_star_h*1e-6))
#     elif theory == 0:
#         #print(V_R,type(V_R))
#         return -constant * ((carrier_n((V_R),N_A,N_D)/(m_star_e*1e-6)) + (carrier_p((V_R),N_A,N_D)/(m_star_h*1e-6)))
#     elif theory == 2:
#         epsilon = permittivity(float(V_R), N_A, N_D)
#         return (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)+ np.real(epsilon))/2)))
# =============================================================================
"""
Source: Design, Analysis, and Performance of a Silicon Photonic Traveling Wave Mach-Zehnder Modulator
        David Patel
"""
# =============================================================================
# def n_effective(voltage, N_A, N_D,theory):  #Theory = 1, synopsis = 0, lorentz = 2
#     V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
#     if theory == 1:
#         return -(5.4e-22 * pow(carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6),1.011) + 1.53e-18 * pow(carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6),0.838))
#     elif theory == 0:
#         #print(V_R,type(V_R))
#         return -(5.4e-22 * pow(carrier_n(V_R,N_A,N_D),1.011) + 1.53e-18 * pow(carrier_p(V_R,N_A,N_D),0.838))
#     elif theory == 2:
#         epsilon = permittivity(float(V_R), N_A, N_D)
#         return (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)+ np.real(epsilon))/2)))
# 
# =============================================================================


def del_phi_eff(V_R, N_A, N_D, LENGTH,theory): #Theory = 1, synopsis = 0, lorentz = 2
    return ((k_0 *abs( n_effective(V_R, N_A, N_D,theory) - n_effective(0, N_A, N_D, theory)) *1e-3 )) % (2*np.pi) #degree phase

"""
Source: Silicon optical modulators  
        G.T. Reed, G. Mashanovich, F.Y. Gardes and D. J. Thomson 
        derived for 1.55 um from Electrooptical  Effects  in  Silicon (taking into consideration changing mobilities)
                                 RICHARD  A.  SOREF
"""
def alpha_eff(voltage,N_A,N_D,theory): #Theory Plasma= 1, synopsis Plasma = 0, lorentz = 2
    u_e, u_h = calculate_mobilities(N_A, N_D, t)
    #print(u_e,u_h)
    V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
    constant = ((e ** 3) * ((wavelength) ** 2))/ (4 * (np.pi**2) * ((c*100)**3) * epsilon_0 * n)
    if theory == 1:
        del_alpha = constant * (carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6)/((m_star_e**2 ) *u_e) + carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)/((m_star_h**2) * u_h)) #/cm 
        alpha_db = 10 * (del_alpha) * np.log10(np.e) #db/cm
        return alpha_db * (LENGTH / 10)
    elif theory == 0:    
        del_alpha = constant * (carrier_n((V_R),N_A,N_D)/((m_star_e**2) *u_e) + carrier_p((V_R),N_A,N_D)/((m_star_h**2) * u_h)) #/cm
        alpha_db = 10 * (del_alpha) * np.log10(np.e) #db/cm
        #print(del_alpha, alpha_db)
        return alpha_db * (LENGTH / 10)
    elif theory == 2:
        epsilon = permittivity(float(V_R), N_A, N_D)
        k_eff = (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)- np.real(epsilon))/2)))
        del_alpha = ( 2*k_0*k_eff )/1e2 #/cm
        alpha_db = 10*(del_alpha)*np.log10(np.e)
        #print(del_alpha, alpha_db)
        return alpha_db * (LENGTH/10)
    elif theory == 3:
        del_alpha =  (8.5e-18 * carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6) + 4e-18 * carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)) #/cm
        alpha_db = 10 * (del_alpha) * np.log10(np.e) #db/cm
        return alpha_db * (LENGTH / 10)
    elif theory == 4:    
        del_alpha =  ((8.5e-18 * carrier_n(V_R,N_A,N_D) + 4e-18 * carrier_p(V_R,N_A,N_D)))
        alpha_db = 10 * (del_alpha) * np.log10(np.e) #db/cm
        return alpha_db * (LENGTH / 10)

type =['exp plasma dispersion','theory plasma dispersion','lorentz', 'exp kramer-kronig','theory kramer-kronig']
def plot_phi_eff(N_A1, N_D1,N_A2,N_D2,N_A3,N_D3,theory):
    if theory == 0 or theory == 4:
        V_R_range =  np.arange(0,-5.1,-0.5) #    V_R_range =  np.arange(0,-10.1,-0.5)
    else:
        V_R_range =  np.arange(0.1,-5.1,-0.05) #    V_R_range =  np.arange(0,-10.1,-0.5)

    
    delta_phi1 = [(del_phi_eff(V_R, N_A1, N_D1,LENGTH,theory) ) for V_R in V_R_range]
    
    delta_phi2 = [(del_phi_eff(V_R, N_A2, N_D2,LENGTH,theory) ) for V_R in V_R_range]
    
    delta_phi3 = [(del_phi_eff(V_R, N_A3, N_D3,LENGTH,theory) ) for V_R in V_R_range]

    
    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range, (delta_phi1), label=rf'Effective del phi $N_A = {N_A1}$ $N_D = {N_D1}$' ,color = 'blue', linestyle = 'dashed')
    plt.plot(V_R_range, (delta_phi2), label=rf'Effective del phi $N_A = {N_A2}$ $N_D = {N_D2}$', color = 'red', linestyle = 'dotted', marker = 'o')
    plt.plot(V_R_range, (delta_phi3), label=rf'Effective del phi $N_A = {N_A3}$ $N_D = {N_D3}$', color = 'green', linestyle = 'dashed')

    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta \phi_{eff}$ [deg/m]', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(f'$\Delta \phi_{{eff}}$ vs. Reverse Bias Voltage type = {type[theory]}', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    
    y_ticks = np.arange(0, 2.1 * np.pi, 0.5 * np.pi)
    y_tick_labels = [f'{tick/np.pi:.1f}π' for tick in y_ticks]

    plt.gca().set_yticks(y_ticks)
    plt.gca().set_yticklabels(y_tick_labels)
    
    plt.legend()
    plt.show()

def plot_n_eff(N_A1, N_D1,N_A2,N_D2,N_A3,N_D3,theory):
    if theory == 0 or theory == 4 : 
        V_R_range =  np.arange(0,-5.1,-0.5)  #  V_R_range =  np.arange(0,-10.1,-0.5)
    else : 
        V_R_range =  np.arange(0,-5.1,-0.05) #    V_R_range =  np.arange(0,-10.1,-0.5)
    

    delta_n_eff_values1 = [((n_effective(V_R, N_A1, N_D1,theory) - n_effective(0, N_A1, N_D1,theory))) for V_R in V_R_range]

    delta_n_eff_values2 = [((n_effective(V_R, N_A2, N_D2,theory) - n_effective(0, N_A2, N_D2,theory))) for V_R in V_R_range]
    delta_n_eff_values3 = [((n_effective(V_R, N_A3, N_D3,theory) - n_effective(0, N_A3, N_D3,theory))) for V_R in V_R_range]

# =============================================================================
#     delta_n_eff_values1 = [((n_effective(V_R, N_A, N_D,0) )) for V_R in V_R_range]
# 
#     delta_n_eff_values3 = [((n_effective(V_R, N_A, N_D,2))) for V_R in V_R_range]
#     delta_n_eff_values2 = [((n_effective(V_R, N_A, N_D,1) )) for V_R in V_R_range]
# =============================================================================

    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range, delta_n_eff_values1, label=rf'Effective del neff $N_A = {N_A1}$ $N_D = {N_D1}$', color = 'blue')
    plt.plot(V_R_range, (delta_n_eff_values2), label=rf'Effective del neff $N_A = {N_A2}$ $N_D = {N_D2}$', color = 'red' , linestyle = 'dotted', marker = 'o')
    plt.plot(V_R_range, (delta_n_eff_values3), label=rf'Effective del neff $N_A = {N_A3}$ $N_D = {N_D3}$', color = 'green')

    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta n_{eff}$ ', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(f'$\Delta n_{{eff}}$ vs. Reverse Bias Voltage type = {type[theory]}', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_alpha_eff(N_A1, N_D1,N_A2,N_D2,N_A3,N_D3,theory):
    if theory == 0 or theory == 4:
        V_R_range =  np.arange(0,-5.1,-0.5)
    else :
        V_R_range =  np.arange(0,-5.1,-0.05)
    
    #delta_alpha1 = [(alpha_eff(V_R, N_A1, N_D1) - alpha_eff(0, N_A1, N_D1)) for V_R in V_R_range]
    #delta_alpha2 = [(alpha_eff(V_R, N_A2, N_D2) - alpha_eff(0, N_A2, N_D2)) for V_R in V_R_range]
    delta_alpha1 = [(alpha_eff(V_R, N_A1, N_D1,theory))for V_R in V_R_range]
    delta_alpha2 = [(alpha_eff(V_R, N_A2, N_D2,theory)) for V_R in V_R_range]
    delta_alpha3 = [(alpha_eff(V_R, N_A3, N_D3,theory)) for V_R in V_R_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range, (delta_alpha1), label=rf'Effective Loss index $N_A = {N_A1}$ $N_D = {N_D1}$', color = 'blue', linestyle = 'solid',linewidth = 4)
    plt.plot(V_R_range, (delta_alpha2), label=rf'Effective loss index $N_A = {N_A2}$ $N_D = {N_D2}$', color = 'red', linestyle = 'dotted',linewidth = 2)
    plt.plot(V_R_range, (delta_alpha3), label=rf'Effective loss index $N_A = {N_A3}$ $N_D = {N_D3}$', color = 'green', linestyle = 'dotted',linewidth = 2)

    plt.yscale('log')
    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta \alpha_{eff}$ [dB] ', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(rf'$\Delta \alpha_{{eff}}$ vs. Reverse Bias Voltage type = {type[theory]}', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,6))

def Vpi_lut(N_A,N_D,theory):
    if theory != 0 and theory != 4:
        V_R = np.arange(-0.1,-5,-0.05)
    else:
        V_R = np.arange(-0.5,-5.1,-0.5)
    lut = []
    for v in V_R:
        row = [v]
        del_n_eff = abs(n_effective(v, N_A, N_D,theory)- n_effective(0, N_A, N_D,theory)) #/cc
        del_phi = ((k_0 *del_n_eff * LENGTH)) % (2*np.pi)
        row.append(del_phi)
        lut.append(row)
        #print(wavelength)
    return lut

def polynomial(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

def Vpi_calc(N_A, N_D, theory):
    lut = Vpi_lut(N_A, N_D, theory)
    V_values, del_phi_values = zip(*lut)

    # Fit a polynomial to the data
    params, _ = curve_fit(polynomial, del_phi_values, V_values)

    # Use the polynomial to estimate Vpi when del_phi = pi
    Vpi = polynomial(np.pi, *params)

    return Vpi

def Lpi(N_A1,N_D1,N_A2,N_D2,N_A3,N_D3,theory):
    if theory != 0:
        V_R = np.arange(-0.1,-10,-0.01)
    else:
        V_R = np.arange(-0.5,-5.1,-0.5)
    lut = []
    for v in V_R:
        row = [v]
        del_n_eff1 = abs(n_effective(v, N_A1, N_D1,theory)- n_effective(0, N_A1, N_D1,theory)) #/cc
        del_n_eff2 = abs(n_effective(v, N_A2, N_D2,theory)- n_effective(0, N_A2, N_D2,theory)) #/cc
        del_n_eff3 = abs(n_effective(v, N_A3, N_D3,theory)- n_effective(0, N_A3, N_D3,theory)) #/cc
        Lpi1 = (wavelength*100)/(2*del_n_eff1 )
        Lpi2 = (wavelength*100)/(2*del_n_eff2 )
        Lpi3 = (wavelength*100)/(2*del_n_eff3 )
        row.append(Lpi1*10)
        row.append(Lpi2*10)
        row.append(Lpi3*10)
        lut.append(row)
    v,Lpi1,Lpi2,Lpi3 = zip(*lut)
    plt.figure(figsize=(10, 6))
    plt.yscale('linear')
    plt.plot(v,Lpi1, color = 'red', label = rf'Lpi $N_A = {N_A1}$ $N_D = {N_D1}$')
    plt.plot(v,Lpi2, color = 'blue', label = rf'Lpi $N_A = {N_A2}$ $N_D = {N_D2}$')
    plt.plot(v,Lpi3, color = 'green', label = rf'Lpi $N_A = {N_A3}$ $N_D = {N_D3}$')
    plt.xlabel('Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel('Lpi (mm)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(f'Lpi vs. Voltage type = {type[theory]}', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()
'''
N_A1 = float(input("Enter the Acceptor concentration (N_A1) in m^-3: "))
N_D1 = float(input("Enter the Donor concentration (N_D1) in m^-3: "))
N_A2 = float(input("Enter the Acceptor concentration (N_A2) in m^-3: "))
N_D2 = float(input("Enter the Donor concentration (N_D2) in m^-3: "))
'''
N_A1 = 5e19
N_D1 = 5e19
N_A2 = 9e19
N_D2 = 1e19
N_A3 = 1e19
N_D3 = 9e19

print()
N_A = N_A1
N_D = N_D1
print(f'N_A = {N_A} N_D = {N_D}')
#print('theory', Vpi_calc(N_A, N_D,1))
print('exp', Vpi_calc(N_A, N_D,0))
#print('lorentz', Vpi_calc(N_A,N_D,2))

print()
N_A = N_A2
N_D = N_D2
print(f'N_A = {N_A} N_D = {N_D}')
#print('theory', Vpi_calc(N_A, N_D,1))
print('exp', Vpi_calc(N_A, N_D,0))
#print('lorentz', Vpi_calc(N_A,N_D,2))

print()
N_A = N_A3
N_D = N_D3
print(f'N_A = {N_A} N_D = {N_D}')
#print('theory', Vpi_calc(N_A, N_D,1))
print('exp', Vpi_calc(N_A, N_D,0))
#print('lorentz', Vpi_calc(N_A,N_D,2))

print()
'''
h_data_by_voltage1 = {}
e_data_by_voltage1 = {}
h_data_by_voltage1,e_data_by_voltage1 =   read_data(N_A1, N_D1)

h_data_by_voltage2 = {}
e_data_by_voltage2 = {}
h_data_by_voltage2,e_data_by_voltage2 =   read_data(N_A2, N_D2)
'''
#Lpi(N_A1, N_D1, N_A2, N_D2, N_A3, N_D3, 0)
#plot_phi_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)
#plot_alpha_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)
#plot_n_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)


Lpi(N_A1, N_D1, N_A2, N_D2, N_A3, N_D3, 0)
#Lpi(N_A1, N_D1, N_A2, N_D2, N_A3, N_D3, 3)
#Lpi(N_A1, N_D1, N_A2, N_D2, N_A3, N_D3, 4)

plot_phi_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)
#plot_phi_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,3)
#plot_phi_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,4)

plot_n_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)
#plot_n_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,3)
#plot_n_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,4)

plot_alpha_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,0)
#plot_alpha_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,3)
#plot_alpha_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,4)

Lpi(N_A1, N_D1, N_A2, N_D2, N_A3, N_D3, 1)
plot_phi_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,1)
plot_n_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,1)
plot_alpha_eff(N_A1, N_D1, N_A2, N_D2,N_A3,N_D3,1)

plot_charge_profile(N_A1, N_D1)
plot_charge_profile(N_A2, N_D2)
plot_charge_profile(N_A3, N_D3)









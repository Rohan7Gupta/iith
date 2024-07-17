# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 07:55:23 2024

@author: Rohan Gupta
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.integrate import quad
#import math

N_A = 5e19
N_D = 5e19

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
    V_bi = (k * t / e) * np.log((N_A * N_D) / n_i**2)
    return V_bi

"""
Source : Electron and Hole Mobilities in Silicon as a Function of Concentration and Temperature
         N.D. Arora; J.R. Hauser; D.J. Roulston
"""
# =============================================================================
# def calculate_mobilities(Nd, Na, T=300):
#     # Constants for electron mobility
#     mu_e_0 = 88
#     T_ref = 300
#     B_e = 7.4e8
#     C_e = 1.26e17
#     alpha_e = -0.57
#     beta_e = -2.33
#     gamma_e = 2.4
#     delta_e = -0.146
# 
#     # Electron mobility calculation
#     mu_e = (mu_e_0 * (T / T_ref) ** alpha_e +
#             (B_e * T ** beta_e) /
#             (1 + 0.88 * ((Nd / (C_e * (T / T_ref) ** gamma_e)) * (T / T_ref) ** delta_e)))
# 
#     # Constants for hole mobility
#     mu_h_0 = 54.3
#     B_h = 1.36e8
#     C_h = 2.35e17
#     alpha_h = -0.57
#     beta_h = -2.33
#     gamma_h = 2.4
#     delta_h = -0.146
# 
#     # Hole mobility calculation
#     mu_h = (mu_h_0 * (T / T_ref) ** alpha_h +
#             (B_h * T ** beta_h) /
#             (1 + 0.88 * ((Na / (C_h * (T / T_ref) ** gamma_h)) * (T / T_ref) ** delta_h)))
# 
#     return mu_e, mu_h
# 
# 
# u_e, u_h = calculate_mobilities(N_D, N_A, t)
# =============================================================================



def depletion_width(V_R, N_A, N_D):
    V_bi = calculate_builtin_voltage(N_A, N_D)
    return np.sqrt(abs(2 * epsilon_s * (V_bi + V_R) * (N_A + N_D) / (e * (N_A * N_D))))

def x_n(V_R, N_A, N_D): # n depletion width
    return depletion_width(V_R, N_A, N_D) * N_A / (N_A + N_D) #meter

def x_p(V_R, N_A, N_D): # p depletion width
    return -depletion_width(V_R, N_A, N_D) * N_D / (N_A + N_D) #meter

"""
 Source: CARRIER CONCENTRATIONS AND DRIFT CURRENTS IN THE DEPLETION REGION OF A p n JUNCTION
         D. K. MAK
"""
def p_x_maj(x, V_R, N_A, N_D): #majority p carrier desity -xp<x<0
    a_p = (e**2)*N_A / (2*epsilon_s*k*t)
    return N_A * np.exp(-a_p * (-x + x_p(V_R, N_A, N_D))**2) # /m3

def n_x_maj(x, V_R, N_A, N_D): #majority n carrier density 0<x<xn
    a_n = (e**2)*N_D / (2*epsilon_s*k*t)
    return N_D * np.exp(-a_n * (x - x_n(V_R, N_A, N_D))**2) # /m3

"""
standard approach (need to det constants experimentally)
"""
# =============================================================================
# # Constants
# k_1 = 8.617e-5  # Boltzmann's constant in eV/K
# n_i = 1.5e10  # Intrinsic carrier concentration in cm^-3
# D_n = 35  # Diffusion coefficient for electrons in cm^2/s
# D_p = 12  # Diffusion coefficient for holes in cm^2/s
# tau_n = 1e-6  # Minority carrier lifetime for electrons in s
# tau_p = 1e-6  # Minority carrier lifetime for holes in s
# 
# 
# # Calculating the thermal voltage
# V_T = k_1 * t / e
# 
# # Calculating diffusion lengths
# L_n = math.sqrt(D_n * tau_n)
# L_p = math.sqrt(D_p * tau_p)
# 
# # Calculating equilibrium minority carrier concentrations
# 
# 
# def p_x_maj(x, V_R, N_A, N_D): #majority p carrier desity -xp<x<0
#     p_p0 = (n_i ** 2) / N_D    
#     return p_p0 * np.exp(V_R/V_T)*np.exp(-x/L_p)
# 
# def n_x_maj(x, V_R, N_A, N_D): #majority n carrier density 0<x<xn
#     n_n0 = (n_i ** 2) / N_A
#     return n_n0 * np.exp(V_R/V_T)*np.exp(-x/L_n)
# =============================================================================

def format_to_4_digits(number):
    # Format the number to a 4-digit string with leading zeros
    formatted_number = f"{number:04}"
    return formatted_number 

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
def n_effective(voltage, N_A, N_D,theory):  #Theory = 1, synopsis = 0, lorentz = 2
    V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
    if theory == 1:
        return -(8.8e-22 * carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6) + 8.5e-18 * pow(carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6),0.8))
    elif theory == 0:
        #print(V_R,type(V_R))
        return -(8.8e-22 * carrier_n(V_R,N_A,N_D) + 8.5e-18 * pow(carrier_p(V_R,N_A,N_D),0.8))
    elif theory == 2:
        epsilon = permittivity(float(V_R), N_A, N_D)
        return (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)+ np.real(epsilon))/2)))

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




def del_phi_eff(V_R, N_A, N_D, LENGTH,theory): #Theory = 1, synopsis = 0, lorentz = 2
    return ((k_0 *abs( n_effective(V_R, N_A, N_D,theory) - n_effective(0, N_A, N_D,theory))*1e6 * LENGTH *1e-3 )*(180/np.pi)) % 360 #degree phase

"""
Source: Silicon optical modulators  
        G.T. Reed, G. Mashanovich, F.Y. Gardes and D. J. Thomson 
        derived for 1.55 um from Electrooptical  Effects  in  Silicon (taking into consideration changing mobilities)
                                 RICHARD  A.  SOREF
"""
def alpha_eff(voltage,N_A,N_D,theory): #Theory = 1, synopsis = 0, lorentz = 2
    V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
    if theory == 1:
        del_alpha =  (8.5e-18 * carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6) + 4e-18 * carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)) #/cm
        alpha_db = 10 * (del_alpha) * np.log(np.e) #db/cm
        return alpha_db * (LENGTH / 10)
    elif theory == 0:    
        del_alpha =  ((8.5e-18 * carrier_n(V_R,N_A,N_D) + 4e-18 * carrier_p(V_R,N_A,N_D)))
        alpha_db = 10 * (del_alpha) * np.log(np.e) #db/cm
        return alpha_db * (LENGTH / 10)
    elif theory == 2:
        epsilon = permittivity(float(V_R), N_A, N_D)
        k_eff = (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)- np.real(epsilon))/2)))
        del_alpha = ( 2*k_0*k_eff )/1e2 #/cm
        alpha_db = 10*(del_alpha)*np.log10(np.e)
        return alpha_db * (LENGTH/10)
"""
Source: Electrooptical  Effects  in  Silicon 
        RICHARD  A.  SOREF, S
"""
#not scaled properly

# =============================================================================
# def alpha_eff(voltage,N_A,N_D,theory): #Theory = 1, synopsis = 0, lorentz = 2
#     V_R = f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))
#     constant = ((e ** 3) * ((wavelength*100) ** 2))/ (4 * (np.pi**2) * ((c*100)**3) * epsilon_0*1e-2 * n)
#     if theory == 1:
#         del_alpha = constant * (carrier_n_theory(float(V_R),N_A*1e6,N_D*1e6)/((m_star_e**2 ) *u_e) + carrier_p_theory(float(V_R),N_A*1e6,N_D*1e6)/((m_star_h**2) * u_h)) #/cm 
#         alpha_db = 10 * (del_alpha) * np.log(np.e) #db/cm
#         return alpha_db * (LENGTH / 10)
#     elif theory == 0:    
#         del_alpha = constant * (carrier_n((V_R),N_A,N_D)/((m_star_e**2) *u_e) + carrier_p((V_R),N_A,N_D)/((m_star_h**2) * u_h)) #/cm
#         alpha_db = 10 * (del_alpha) * np.log(np.e) #db/cm
#         print(del_alpha, alpha_db)
#         return alpha_db * (LENGTH / 10)
#     elif theory == 2:
#         epsilon = permittivity(float(V_R), N_A, N_D)
#         k_eff = (np.sqrt(abs((np.sqrt(np.real(epsilon)**2 + np.imag(epsilon)**2)- np.real(epsilon))/2)))
#         del_alpha = ( 2*k_0*k_eff )/1e2 #/cm
#         alpha_db = 10*(del_alpha)*np.log10(np.e)
#         print(del_alpha, alpha_db)
#         return alpha_db * (LENGTH/10)
# =============================================================================

def plot_phi_eff(N_A1, N_D1):
    V_R_range1 =  np.arange(0,-5.1,-0.5) #    V_R_range =  np.arange(0,-10.1,-0.5)

    
    delta_phi1 = [(del_phi_eff(V_R, N_A, N_D,LENGTH,0) - del_phi_eff(0, N_A, N_D,LENGTH,0) ) for V_R in V_R_range1]
    
    V_R_range =  np.arange(0,-5.1,-0.005) #    V_R_range =  np.arange(0,-10.1,-0.5)

    delta_phi3 = [(del_phi_eff(V_R, N_A, N_D,LENGTH,2) - del_phi_eff(0, N_A, N_D,LENGTH,2) ) for V_R in V_R_range]
    
    delta_phi2 = [(del_phi_eff(V_R, N_A, N_D,LENGTH,1) - del_phi_eff(0, N_A, N_D,LENGTH,1) ) for V_R in V_R_range]

    
    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range1, (delta_phi1), label='Effective del phi synopsis' ,color = 'blue', linestyle = 'dashed')
    plt.plot(V_R_range, (delta_phi2), label='Effective del phi literature', color = 'red', linestyle = 'dashed')
    plt.plot(V_R_range, (delta_phi3), label='Effective del phi lorentz', color = 'green', linestyle = 'dashed')

    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta \phi_{eff}$ [deg/mm]', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(rf'$\Delta \phi_{{eff}}$ vs. Reverse Bias Voltage $N_A = {N_A}$ $N_D = {N_D}$', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_n_eff(N_A, N_D):
    V_R_range1 =  np.arange(0,-5.1,-0.5)  #  V_R_range =  np.arange(0,-10.1,-0.5)
    V_R_range =  np.arange(0,-5.1,-0.005) #    V_R_range =  np.arange(0,-10.1,-0.5)


    delta_n_eff_values1 = [((n_effective(V_R, N_A, N_D,0) - n_effective(0, N_A, N_D,0))) for V_R in V_R_range1]

    delta_n_eff_values3 = [((n_effective(V_R, N_A, N_D,2) - n_effective(0, N_A, N_D,2))) for V_R in V_R_range]
    delta_n_eff_values2 = [((n_effective(V_R, N_A, N_D,1) - n_effective(0, N_A, N_D,1))) for V_R in V_R_range]

# =============================================================================
#     delta_n_eff_values1 = [((n_effective(V_R, N_A, N_D,0) )) for V_R in V_R_range]
# 
#     delta_n_eff_values3 = [((n_effective(V_R, N_A, N_D,2))) for V_R in V_R_range]
#     delta_n_eff_values2 = [((n_effective(V_R, N_A, N_D,1) )) for V_R in V_R_range]
# =============================================================================


    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range1, delta_n_eff_values1, label='Effective Refractive Index synoposis', color = 'blue')
    plt.plot(V_R_range, (delta_n_eff_values2), label='Effective Refractive Index literature', color = 'red')
    plt.plot(V_R_range, (delta_n_eff_values3), label='Effective Refractive Index lorentz', color = 'green')

    
    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta n_{eff}$ ', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(rf'$\Delta n_{{eff}}$ vs. Reverse Bias Voltage $N_A = {N_A}$ $N_D = {N_D}$', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_alpha_eff(N_A, N_D):
    V_R_range1 =  np.arange(0,-5.1,-0.5)
    V_R_range =  np.arange(0,-5.1,-0.005)
    
    #delta_alpha1 = [(alpha_eff(V_R, N_A1, N_D1) - alpha_eff(0, N_A1, N_D1)) for V_R in V_R_range]
    #delta_alpha2 = [(alpha_eff(V_R, N_A2, N_D2) - alpha_eff(0, N_A2, N_D2)) for V_R in V_R_range]
    delta_alpha1 = [(alpha_eff(V_R, N_A, N_D,0))for V_R in V_R_range1]
    delta_alpha2 = [(alpha_eff(V_R, N_A, N_D,1)) for V_R in V_R_range]
    delta_alpha3 = [(alpha_eff(V_R, N_A, N_D,2)) for V_R in V_R_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(V_R_range1, (delta_alpha1), label='Effective Loss index synopsis', color = 'blue', linestyle = 'solid',linewidth = 4)
    plt.plot(V_R_range, (delta_alpha2), label='Effective loss index literature', color = 'red', linestyle = 'dotted',linewidth = 2)
    plt.plot(V_R_range, (delta_alpha3), label='Effective loss index lorentz', color = 'green', linestyle = 'dotted',linewidth = 2)


    plt.xlabel('Reverse Bias Voltage (V)', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.ylabel(r' $\Delta \alpha_{eff}$ [dB] ', fontdict={'family': 'Times New Roman', 'size': 12})
    plt.title(rf'$\Delta \alpha_{{eff}}$ vs. Reverse Bias Voltage $N_A = {N_A}$ $N_D = {N_D}$', fontdict={'family': 'Times New Roman', 'size': 14})
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.figure(figsize=(10,6))


def calculate_junction_capacitance(N_A,N_D):
    voltages_range =  np.arange(-0.5,-5.1,-0.5)
    cv = []
    
    voltages = [f"{voltage:.1f}".rstrip('0').rstrip('.') if voltage % 1 != 0 else str(int(voltage))  for voltage in voltages_range]
    total_charges = [e*(carrier_p(V,N_A,N_D) + carrier_n(V,N_A,N_D)) for V in voltages] #couloumb/cc
    capacitances = np.gradient(total_charges,voltages_range) #F/cc
    #plt.plot(abs(voltages_range),total_charges, marker='o')
    #print (total_charges,voltages_range,capacitances)
    cv.append(voltages_range)
    cv.append(capacitances * length * 1e2) #F/cm2
    return cv

def capacitances_theory(V,N_A,N_D) :
    V_bi = calculate_builtin_voltage(N_A, N_D)
    return np.sqrt((e*epsilon_s*1e-2*N_A*N_D)/(2*(N_A+N_D)*(V_bi - V)))

def capacitances_extract(V):
    return (1.718e-9 * V ** 8 - 3.719e-8 * V ** 7 + 3.29e-7 * V ** 6 - 1.531e-6 * V ** 5 +
            4.022e-6 * V ** 4 - 6.027e-6 * V ** 3 + 5.255e-6 * V ** 2 - 5.255e-6 * V + 2.65e-6)
    
def plot_cap(N_A,N_D):  
    cv = calculate_junction_capacitance(N_A,N_D) #dq/dv
    capacitance = [capacitances_theory(V,N_A,N_D)for V in cv[0]] #formula
    #capacitance_extract = [capacitances_extract(abs(V))for V in cv[0]] #formula
    plt.figure(figsize=(10, 6))
    plt.plot(abs(cv[0]),cv[1], marker='o', color = 'red', label = 'capacitance experimental')
    plt.plot(abs(cv[0]),capacitance, color = 'blue', label = 'capacitance theory')
    #plt.plot(abs(cv[0]),capacitance_extract, color = 'green', label = 'capacitance extract')

    #plt.yscale('log')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Capacitance (F/cm\u00b2)')
    plt.title(f'Junction Capacitance vs. Voltage $N_A = {N_A}$ $N_D = {N_D}$')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    

# =============================================================================
# N_A1 = float(input("Enter the Acceptor concentration (N_A1) in m^-3: "))
# N_D1 = float(input("Enter the Donor concentration (N_D1) in m^-3: "))
# N_A2 = float(input("Enter the Acceptor concentration (N_A2) in m^-3: "))
# N_D2 = float(input("Enter the Donor concentration (N_D2) in m^-3: "))
# 
# =============================================================================


def Vpi_lut(N_A,N_D,theory):
    V_R = np.arange(0,-5.1,-0.5)
    lut = []
    for v in V_R:
        row = [v]
        del_n_eff = abs(n_effective(v, N_A, N_D,theory)- n_effective(0, N_A, N_D,theory)) #/cc
        del_phi = (k_0 * del_n_eff*1e6 *LENGTH*1e-3  )% (2*np.pi)
        row.append(del_phi)
        lut.append(row)
        #print(wavelength)
    return lut

def Vpi_calc(N_A,N_D,theory):
    lut = Vpi_lut(N_A, N_D,theory)
    V_values, del_phi_values = zip(*lut)

    # Use interpolation to find V when del_phi = pi
    interp_func = interp1d(del_phi_values, V_values, kind='linear', fill_value='extrapolate')
    Vpi= interp_func(np.pi)

    return Vpi




h_data_by_voltage1 = {}
e_data_by_voltage1 = {}
h_data_by_voltage1,e_data_by_voltage1 =   read_data(N_A, N_D)

print(f'N_A = {N_A} N_D = {N_D}')

print('theory', Vpi_calc(N_A, N_D,1))
print('exp', Vpi_calc(N_A, N_D,0))
print('lorentz', Vpi_calc(N_A,N_D,2))

plot_cap(N_A,N_D)
plot_phi_eff(N_A, N_D)
plot_n_eff(N_A, N_D)
plot_alpha_eff(N_A, N_D)
plot_charge_profile(N_A, N_D)
#plot_charge_profile(N_A2, N_D2)
Vpi_calc(N_A, N_D,0)

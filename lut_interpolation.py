import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Step 1: Read the LUT.csv file
data = pd.read_csv('LUT.csv')

# Assuming the first column is 'V_R' and subsequent columns are for different doping values
V_R = data.iloc[:, 0].values
doping_values = np.array([float(col.split('=')[1]) for col in data.columns[1:]])
doping_data = data.iloc[:, 1:].values

# Step 2: Interpolate to find Vpi for each doping value
Vpi_values = []

for col in range(doping_data.shape[1]):
    del_phi_values = doping_data[:, col]
    # Create interpolation function
    interpolation_function = interp1d(del_phi_values, V_R, kind='linear', fill_value='extrapolate')
    # Find Vpi where del_phi = pi
    Vpi = interpolation_function(np.pi)
    if 0<=Vpi<=50 : 
        Vpi_values.append(Vpi)
    else:
        Vpi_values.append(-1)

# Step 3: Plot Vpi vs N_doping /cc
plt.figure(figsize=(40, 24))
plt.plot(doping_values, Vpi_values, marker='o', linestyle='-')
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('N_doping [/cc]')
plt.ylabel('Vpi')
plt.title('Vpi vs N_doping')
plt.grid(True, which="both", ls="--")
plt.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Reference Line at -5V')
plt.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Reference Line at -5V')
plt.axhline(y=0, color='magenta', linestyle='--', linewidth=2, label='Reference Line at -5V')


plt.show()

# Save Vpi values less than 5V with corresponding doping concentrations to a file
with open('Vpi_less_than_5V.txt', 'w') as f:
    for doping, Vpi in zip(doping_values, Vpi_values):
        if 0< Vpi <= 5:
            f.write(f"Doping concentration: {doping:.2e} /cc, Vpi: {Vpi:.10f} V\n")

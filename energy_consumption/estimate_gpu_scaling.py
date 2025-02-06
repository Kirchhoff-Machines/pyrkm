import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Read the CSV file
df = pd.read_csv('./gpu_energy_power_analysis.csv')

# Extract data
Nh = df['Nh'].values
time = df['Time (s)'].values
energy_gpu = df['GPU Energy (J)'].values
energy_cpu = df['CPU Energy (J)'].values
energy = energy_gpu + energy_cpu

# Fit a polynomial to the time data
degree = 2  # You can change the degree of the polynomial
poly_time = Polynomial.fit(np.log10(Nh), np.log10(time), degree)

# Fit a polynomial to the energy data
poly_energy = Polynomial.fit(np.log10(Nh), np.log10(energy), degree)

# Define larger values of Nh for extrapolation
Nh_large = np.logspace(np.log10(Nh[-1]), 6, num=10, base=10).astype(int)

# Combine original and larger Nh values
Nh_combined = np.concatenate((Nh, Nh_large))

# Estimate time and energy for combined Nh values
estimated_time_combined = 10**poly_time(np.log10(Nh_combined))
estimated_energy_combined = 10**poly_energy(np.log10(Nh_combined))

# Print the estimated values
print("Estimated Time and Energy for larger values of Nh:")
for nh, t, e in zip(Nh_large, estimated_time_combined[len(Nh):], estimated_energy_combined[len(Nh):]):
    print(f"Nh: {nh}, Estimated Time: {t:.6e} s, Estimated Energy: {e:.6e} J")

# Get the polynomial coefficients
coeffs_time = poly_time.convert().coef
coeffs_energy = poly_energy.convert().coef

# Format the polynomial equations as table-like strings with rounded coefficients
poly_time_str = "\n".join([f"c{i}: {coeff:.2e}" for i, coeff in enumerate(coeffs_time)])
poly_energy_str = "\n".join([f"c{i}: {coeff:.2e}" for i, coeff in enumerate(coeffs_energy)])

# Format the functional form using c1, c2, etc.
func_time_str = " + ".join([f"c{i}*log10(Nh)^{i}" for i in range(len(coeffs_time))])
func_energy_str = " + ".join([f"c{i}*log10(Nh)^{i}" for i in range(len(coeffs_energy))])

# Plot the original data and the combined fit/extrapolated values
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(Nh, time, label='Original Data', zorder=10)
plt.plot(Nh_combined, estimated_time_combined, '--', label=f'Fit:\n{poly_time_str}\n{func_time_str}', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Time (s)')
plt.title('Time vs Nh')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(Nh, energy, label='Original Data', zorder=5)
plt.plot(Nh_combined, estimated_energy_combined, '--', label=f'Fit:\n{poly_energy_str}\n{func_energy_str}', color='red')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Energy (J)')
plt.title('Energy vs Nh')
plt.legend()

plt.tight_layout()
plt.savefig('gpu_energy_power_analysis_extrapolation.png')
plt.show()


# Save the estimated points to a CSV file
estimated_df = pd.DataFrame({
    'Nh': Nh_combined,
    'Estimated Time (s)': estimated_time_combined,
    'Estimated Energy (J)': estimated_energy_combined
})
estimated_df.to_csv('gpu_estimated_points.csv', index=False)
print("Estimated points saved to gpu_estimated_points.csv")
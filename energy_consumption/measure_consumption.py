import torch
import sys
import pyRAPL
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # For CSV export
import gc  # For garbage collection

# Initialize pyRAPL
pyRAPL.setup()

def get_repeat_value(x):
    if x < 1000:
        return 10000
    elif x < 10000:
        return 500
    else:
        return 10

def measure_energy_power(vector_sizes, vector_repeats, num_runs=20, warmup_runs=5, force_cpu=False):
    results = []

    # Determine the device to use
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    if device.type == 'cuda':
        print("ERROR: Using CUDA device for matmul, but measuring CPU consumption.")
        sys.exit(1)

    for size, repeat_operations in zip(vector_sizes, vector_repeats):
        total_time = 0
        total_energy = 0
        print(f"\n********************\nMeasuring energy and power for Nh = {size} on {device}...")

        for _ in range(warmup_runs):
            # Warm-up runs to stabilize CPU
            matrix = torch.rand(size, size, device=device)
            vector = torch.rand(size, device=device)
            result = torch.matmul(matrix, vector)

        for _ in range(num_runs):
            while True:
                # Create random matrix and vector using PyTorch
                matrix = torch.rand(size, size, device=device)
                vector = torch.rand(size, device=device)

                # Measure energy and time
                meter = pyRAPL.Measurement('matrix_vector_product')
                meter.begin()

                start_time = time.perf_counter()
                for _ in range(repeat_operations):
                    result = torch.matmul(matrix, vector)  # Matrix-vector multiplication
                end_time = time.perf_counter()

                meter.end()

                # Accumulate results
                elapsed_time = end_time - start_time
                energy_consumed = sum(meter.result.pkg)  # Sum energy across all sockets

                # Clean up to free memory
                del matrix, vector, result
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Force garbage collection
                gc.collect()

                if energy_consumed > 0:
                    total_time += elapsed_time / repeat_operations
                    total_energy += energy_consumed / repeat_operations
                    break
                else:
                    print(f"Warning: Measured energy consumption is zero. Retrying Nh={size}...")

        # Average results over all runs
        avg_time = total_time / num_runs
        avg_energy = total_energy * 10**-6 / num_runs  # Convert microJoules to Joules
        avg_power = avg_energy / avg_time  # Power in watts

        results.append({
            'Nh': size,
            'Time (s)': avg_time,
            'Energy (J)': avg_energy,
            'Power (W)': avg_power
        })

    return results, device

# Define vector sizes to test
vector_sizes = np.logspace(1, 4.7, num=20, base=10).astype(int).tolist()
vector_repeats = [get_repeat_value(x) for x in vector_sizes]

# Perform measurements
results, device = measure_energy_power(vector_sizes, vector_repeats, force_cpu=True)

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save results to a CSV file
csv_filename = f'{device}_energy_power_analysis.csv'
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

# Extract data for plotting
sizes = df_results['Nh']
times = df_results['Time (s)']
energies = df_results['Energy (J)']
powers = df_results['Power (W)']

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(sizes, times, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Time (s)')
plt.title('Time vs Nh')

plt.subplot(1, 3, 2)
plt.plot(sizes, energies, marker='o', color='orange')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Energy (J)')
plt.title('Energy vs Nh')

plt.subplot(1, 3, 3)
plt.plot(sizes, powers, marker='o', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Power (W)')
plt.title('Power vs Nh')

plt.tight_layout()

# Save the figure as a PNG file
png_filename = f'{device}_energy_power_analysis.png'
plt.savefig(png_filename)
plt.show()

# Print results
for _, row in df_results.iterrows():
    print(f"Nh: {row['Nh']}, Time: {row['Time (s)']:.4f} s, Energy: {row['Energy (J)']:.4f} J, Power: {row['Power (W)']:.4f} W")
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import pandas as pd
import pyRAPL  # For CPU energy measurement

# Initialize pyRAPL
pyRAPL.setup()

def get_gpu_power():
    """Get the current GPU power draw in watts."""
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        # Parse the first GPU's power usage (assuming a single GPU setup)
        power = float(output.strip().split('\n')[0])
        return power
    except Exception as e:
        print(f"Error reading GPU power: {e}")
        return 0.0

def measure_energy_power(vector_sizes, vector_repeats, num_runs=10, warmup_runs=5):
    results = []

    for size, repeat_operations in zip(vector_sizes, vector_repeats):
        total_time = 0
        total_gpu_energy = 0
        total_cpu_energy = 0 
        total_gpu_power = 0
        print(f"\n********************\nMeasuring energy and power for Nh = {size}...")

        for _ in range(warmup_runs):
            # Warm-up runs to stabilize GPU
            matrix = torch.rand(size, size).cuda()
            vector = torch.rand(size).cuda()
            result = torch.matmul(matrix, vector)
            torch.cuda.synchronize()

        for _ in range(num_runs):
            # Create random matrix and vector using PyTorch
            matrix = torch.rand(size, size).cuda()
            vector = torch.rand(size).cuda()

            # Measure time and energy
            meter = pyRAPL.Measurement('matrix_vector_product')
            meter.begin()

            start_time = time.perf_counter()

            for _ in range(repeat_operations):
                result = torch.matmul(matrix, vector)  # Matrix-vector multiplication

            torch.cuda.synchronize()
            end_time = time.perf_counter()
            meter.end()

            gpu_power = get_gpu_power()  # Final power draw in watts

            # Calculate elapsed time and energy
            elapsed_time = end_time - start_time
            gpu_energy_consumed = gpu_power * elapsed_time  # Energy in joules
            cpu_energy_consumed = sum(meter.result.pkg) * 10**-6  # Convert microJoules to Joules

            # Clean up to free memory
            del matrix, vector, result
            torch.cuda.empty_cache()

            total_time += elapsed_time / repeat_operations
            total_gpu_energy += gpu_energy_consumed / repeat_operations
            total_cpu_energy += cpu_energy_consumed / repeat_operations
            total_gpu_power += gpu_power / repeat_operations

        # Average results over all runs
        avg_time = total_time / num_runs
        avg_gpu_energy = total_gpu_energy / num_runs
        avg_cpu_energy = total_cpu_energy / num_runs
        avg_gpu_power = total_gpu_power / num_runs
        avg_power = avg_cpu_energy / avg_time + avg_gpu_power  # Power in watts

        results.append({
            'Nh': size,
            'Time (s)': avg_time,
            'GPU Energy (J)': avg_gpu_energy,
            'CPU Energy (J)': avg_cpu_energy,
            'Power (W)': avg_power
        })

    return results

# Define vector sizes to test
vector_sizes = np.logspace(1, 4.3, num=20, base=10).astype(int).tolist()
def get_repeat_value(x):
    if x < 1000:
        return 10000
    elif x < 10000:
        return 500
    else:
        return 10
vector_repeats = [get_repeat_value(x) for x in vector_sizes]

# Perform measurements
results = measure_energy_power(vector_sizes, vector_repeats)

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save results to a CSV file
csv_filename = 'gpu_energy_power_analysis.csv'
df_results.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")

# Extract data for plotting
sizes = df_results['Nh']
times = df_results['Time (s)']
gpu_energies = df_results['GPU Energy (J)']
cpu_energies = df_results['CPU Energy (J)']
powers = df_results['Power (W)']

# Plot results
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(sizes, times, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nh')
plt.ylabel('Time (s)')
plt.title('Time vs Nh')

plt.subplot(1, 3, 2)
plt.plot(sizes, gpu_energies + cpu_energies, marker='o', color='orange', label='Total')
plt.plot(sizes, gpu_energies, '--', color='red', label='GPU part')
plt.plot(sizes, cpu_energies, '--', color='blue', label='CPU part')
plt.legend()
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
plt.savefig('gpu_cpu_total_energy_power_analysis.png')
plt.show()

# Print results
for _, row in df_results.iterrows():
    print(f"Nh: {row['Nh']}, Time: {row['Time (s)']:.4f} s, GPU Energy: {row['GPU Energy (J)']:.4f} J, CPU Energy: {row['CPU Energy (J)']:.4f} J, Power: {row['Power (W)']:.4f} W")
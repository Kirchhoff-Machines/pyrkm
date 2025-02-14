# Energy Consumption

## Overview

This documentation provides an overview of the `energy_consumption` [submodule](https://github.com/Kirchhoff-Machines/rkm_energy) and its components. The submodule is designed to handle various measurements related to energy consumption and to produce the data reported in the paper XXX.

For the purpose of the study XXX, we consider only the energy consumed by: (i) the **circuit structure** of the RKM, which can be measured analytically using Kirchhoff's laws, and (ii) the **matrix multiplication on a computer**, the simplest operation behind the RKM, for which we introduce specific tools in this submodule.

## Tools and Limitations

To capture the energy consumed by the computer, we use the following tools:
* `pyRAPL`: a [Python package](https://pyrapl.readthedocs.io/en/latest/) that uses Intel's “Running Average Power Limit” (RAPL) technology to estimate CPU power consumption. This technology is available on Intel CPUs since the Sandy Bridge generation.
* `nvidia-smi`: the [standard](https://developer.nvidia.com/system-management-interface) NVIDIA System Management Interface to measure GPU energy consumption.

This submodule can measure the energy consumption of both the CPU and the GPU, but it is limited to Intel CPUs and NVIDIA GPUs.

## Content of the Submodule

In the `src/` directory, you will find several files to analyze the energy consumption of matrix multiplication as a function of the number of hidden nodes `Nh` of the machine. Running these codes will produce images in `out_png` and store data in `out_csv`. We include the images and data used to produce the results in XXX.

The specific files for the analysis are:

#### `measure_consumption.py`
Measures the energy consumption and power usage of matrix-vector multiplications using PyTorch on a CPU, varying the size of the vectors. It plots and saves the results as a CSV file and a PNG image.

#### `nvidiasmi_gpu.py`
Measures the energy consumption and power usage of matrix-vector multiplications using PyTorch on a GPU, varying the size of the vectors. It also measures CPU energy if enabled, and plots and saves the results as a CSV file and a PNG image.

#### `estimate_cpu_scaling.py`
Reads energy and time data from a CSV file, fits polynomial models to the data, and extrapolates to predict values for larger sizes (`Nh`). It plots the original and extrapolated data, saves the plot as a PNG file, and exports the estimated values to a new CSV file.

#### `estimate_gpu_scaling.py`
Reads GPU and CPU energy data from a CSV file, fits polynomial models to the time and total energy data, and extrapolates to predict values for larger sizes (`Nh`). It plots the original and extrapolated data, saves the plot as a PNG file, and exports the estimated values to a new CSV file.

## Conclusion

The `energy_consumption` submodule provides tools to measure and analyze the energy consumption of matrix multiplications on both CPUs and GPUs. By using `pyRAPL` and `nvidia-smi`, we can capture detailed energy usage data, which is essential for understanding the efficiency of the RKM's operations. The provided scripts facilitate the analysis and visualization of this data, enabling further research and optimization.

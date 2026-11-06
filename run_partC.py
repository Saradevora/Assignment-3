import subprocess
import numpy as np
import pandas as pd
import os

sizes = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
threads_list = [1, 2, 4, 8, 16, 24]
repetitions = 5

raw_data = []

subprocess.run([r"C:\w64devkit\bin\gcc.exe", "-O2", "-fopenmp", "partC_parallel.c", "-o", "partC_parallel.exe"], check=True)

print("Running sequential baseline...")
for n in sizes:
    times = []
    for _ in range(repetitions):
        res = subprocess.check_output(["partC_parallel.exe", str(n), "1", "seq"]).decode('utf-8').strip()
        _, _, _, time_ms, checksum = res.split(',')
        times.append(float(time_ms))
    raw_data.append({
        'n': n, 'threads': 1, 'mode': 'seq',
        'median_ms': np.median(times),
        'iqr_ms': np.percentile(times, 75) - np.percentile(times, 25),
        'checksum': float(checksum)
    })

print("Running OpenMP parallel benchmarks...")
for p in threads_list:
    for n in sizes:
        times = []
        for _ in range(repetitions):
            res = subprocess.check_output(["partC_parallel.exe", str(n), str(p), "omp"]).decode('utf-8').strip()
            _, _, _, time_ms, checksum = res.split(',')
            times.append(float(time_ms))
        raw_data.append({
            'n': n, 'threads': p, 'mode': 'omp',
            'median_ms': np.median(times),
            'iqr_ms': np.percentile(times, 75) - np.percentile(times, 25),
            'checksum': float(checksum)
        })

df = pd.DataFrame(raw_data)
df.to_csv("results_partC.csv", index=False)
print("Saved results to results_partC.csv")
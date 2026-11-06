import subprocess
import numpy as np
import pandas as pd

sizes = [10, 50, 100, 200, 300, 400, 500, 600, 800, 1000]
threads_list = [1, 2, 4, 8, 16, 24]
repetitions = 5

raw_runs = []

subprocess.run([r"C:\w64devkit\bin\gcc.exe", "-O2", "-fopenmp", "partC_parallel.c", "-o", "partC_parallel.exe"], check=True)

print("Running sequential baseline...")
for n in sizes:
    for rep in range(repetitions):
        res = subprocess.check_output(["partC_parallel.exe", str(n), "1", "seq"]).decode('utf-8').strip()
        _, _, _, time_ms, checksum = res.split(',')
        raw_runs.append({
            'n': n, 'threads': 1, 'mode': 'seq', 'rep': rep + 1,
            'time_ms': float(time_ms), 'checksum': float(checksum)
        })

print("Running OpenMP parallel benchmarks...")
for p in threads_list:
    for n in sizes:
        for rep in range(repetitions):
            res = subprocess.check_output(["partC_parallel.exe", str(n), str(p), "omp"]).decode('utf-8').strip()
            _, _, _, time_ms, checksum = res.split(',')
            raw_runs.append({
                'n': n, 'threads': p, 'mode': 'omp', 'rep': rep + 1,
                'time_ms': float(time_ms), 'checksum': float(checksum)
            })

df_raw = pd.DataFrame(raw_runs)
df_raw.to_csv("results_partC_raw.csv", index=False)

df_summary = df_raw.groupby(['n', 'threads', 'mode']).agg(
    median_ms=('time_ms', 'median'),
    iqr_ms=('time_ms', lambda x: np.percentile(x, 75) - np.percentile(x, 25)),
    checksum=('checksum', 'first')
).reset_index()
df_summary.to_csv("results_partC.csv", index=False)

print("Saved raw runs to results_partC_raw.csv and summary to results_partC.csv!")
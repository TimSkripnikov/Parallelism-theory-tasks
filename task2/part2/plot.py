import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")

threads = df["Threads"].to_list()
ideal_speedup = df["Threads"].to_list()
speedup = df["Speedup"].to_list()

plt.figure(figsize=(5, 5))
plt.plot(threads, speedup, marker='o', label='speedup')
plt.plot(threads, ideal_speedup, marker='o')
plt.xlabel("Number of threads")
plt.ylabel("SpeedUp")
plt.xticks(threads)
plt.tight_layout()
plt.axis("equal")
plt.grid()
plt.savefig("speedup_plot.png")
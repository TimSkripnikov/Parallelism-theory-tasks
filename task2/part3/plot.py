import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results_2.csv")

threads = [1, 2, 4, 7, 8, 16, 20, 40]
ideal_speedup = [1, 2, 4, 7, 8, 16, 20, 40]

speedup_20k = [1.0] + [df.iloc[0, column] for column in range(2, 15, 2)]

plt.figure(figsize=(5, 5))
plt.plot(threads, speedup_20k, marker='o', label='n=1000')
plt.plot(threads, ideal_speedup, 'r--', label='perfect acceleration')

plt.xlabel("Number of threads")
plt.ylabel("SpeedUp")
plt.xticks(threads)
plt.tight_layout()
plt.axis("equal")
plt.grid()

plt.savefig("speedup_plot_2.png")


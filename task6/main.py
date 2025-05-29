import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = np.loadtxt('output.txt')

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Temperature')
plt.title('Heat map of the temperature distribution')
plt.show()
plt.savefig("map.png")
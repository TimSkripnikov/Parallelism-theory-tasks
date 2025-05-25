import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = np.loadtxt('output.txt')

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Температура')
plt.title('Тепловая карта распределения температуры')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
plt.savefig("map.png")
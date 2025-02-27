import math
import time
import yaml
import matplotlib.pyplot as plt
import numpy as np


# Считывание данных из файла config.
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    db_xmin = config['database']['xmin']
    db_step = config['database']['step']
    db_xmax = config['database']['xmax']
    db_a = config['database']['a']
    db_b = config['database']['b']
    db_c = config['database']['c']

x = []
b = []
# Цикл для вычисления функции.
while db_xmin < db_xmax:
    y = 2 * db_xmin + ((db_a * math.pow(math.sin(db_b * db_xmin + db_c), 2)) / (3 + db_xmin))
    b.append(y)
    x.append(db_xmin)
    db_xmin += db_step
    with open("result.txt", "w") as file:
        for c in b:
            file.write(str(c))
            file.write('\n')
x_1 = np.arange(-8.56, 100, 1)
print("y =", x_1 * 2 + (db_a * (np.sin(db_b * x_1 + db_c) ** 2) / (3 + x_1)))

plt.plot(x, b, color='green', marker='o', markersize=1)
plt.show()


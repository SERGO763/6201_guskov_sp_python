import math
import yaml
import matplotlib.pyplot as plt

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
while db_xmin < db_xmax:
    y = 2 * db_xmin + (db_a * math.pow(2, math.sin(db_b * db_xmin + db_c))) / (3 + db_xmin)
    db_xmin += db_step
    x.append(db_xmin)
    b.append(y)
    with open("result.txt", "w") as file:
        for c in b:
            print(c)
            file.write(str(c))
            file.write('\n')
plt.plot(x, b, color='green', marker='o', markersize=1)
plt.show()



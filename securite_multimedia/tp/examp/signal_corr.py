import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

mu, sigma = 0, 0.1  # mean and standard deviation
l = 10

y = list(np.random.normal(mu, sigma, l))
y_2 = []
y_2.extend(y * 2)
print(y_2)


corr = np.correlate(y_2, y_2, mode='same')
x = [i for i in range(len(y_2))]

print(len(corr))

plt.plot(x, corr)
plt.show()

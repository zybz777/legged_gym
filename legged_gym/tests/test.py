import matplotlib.pyplot as plt
import numpy as np


def Pz(H, t):
    return H * (t - 1 / (2 * np.pi) * np.sin(2 * np.pi * t))


def Pdz(H, t):
    return H * (1 - np.cos(2 * np.pi * t))


t = np.linspace(-1, 1, 100)
print(t)
T = 1
H = 0.05
y = []
dy = []
for dt in t:
    y.append(Pz(H, dt))
    dy.append(Pdz(H, dt))

plt.plot(t, y)
plt.plot(t, dy)
plt.show()

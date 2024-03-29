import numpy as np
import matplotlib.pyplot as plt


def phi(x, theta):
    k = 1 / theta
    return (k * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.square(x * k))


if __name__ == '__main__':
    t = np.linspace(0, 10, 500)
    t_foot = np.sin(2 * np.pi * t)
    theta = 0.5
    phi_foot = phi(t_foot, theta)
    C_foot = (phi(t_foot, theta) * (1 - phi(t_foot - 0.5, theta))
              + phi(t_foot - 1, theta)) * (1 - phi(t_foot - 1.5, theta))
    plt.figure(0)
    plt.plot(t)
    plt.figure(1)
    plt.plot(t_foot)
    plt.figure(2)
    plt.plot(phi_foot)
    plt.figure(3)
    plt.plot(C_foot)
    plt.show()
    print(t)
    print(t_foot)
    pass

"""
Exercise 2-b
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

studentId = "    -No.521021911101"

PI = np.pi
fs = 5000
T = 1 / fs

f0 = 1000 / (2 * PI)


fig, (plt1, plt2) = plt.subplots(2, 1)


def x(t):
    return np.sin(2 * PI * f0 * t)


# plot x(t) figure
t = np.arange(0, 3 / f0, 1e-5)
plt1.plot(t, x(t), label="$x(t)$")

# plot x(nT) figure
nT = np.arange(0, 3 / f0, T)
plt1.plot(nT, x(nT), "go", label="$x(nT)$", markerfacecolor="none")

plt1.axis([0, 3 / f0, -1.2, 1.2])
plt1.axhline(0, color="black", linewidth=0.2)  # draw x axis
plt1.set_xlabel("t")
plt1.set_ylabel("$x(t), x(nT)$")
plt1.set_title("3 periods of $x(t)$ and $x(nT)$" + studentId)
plt1.legend()

n = np.arange(0, 3 / f0 / T, 1)
plt2.stem(n, x(n * T), label="$x[n]$", basefmt="C7-")

plt2.axis([0, 3 / f0 / T, -1.2, 1.2])
plt2.set_xlabel("n")
plt2.set_ylabel("$x[n]$")
plt2.set_title("'3 periods' of $x[n]$" + studentId)
plt2.legend()

plt.show()

# plot n ~ x[n] figure
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

num_of_series = 20
n = np.arange(0, num_of_series, 1)
f0_series = 20 * n

for f0 in f0_series:
    f = f0 * np.ones(num_of_series)
    z = np.sin(2 * PI * f0 * n * T)
    ax.plot(n, f, z)

ax.set_title("$n$ ~ $x[n]$ figure" + studentId)
ax.set_xlabel("$n$")
ax.set_ylabel("$f$")
ax.set_zlabel("$x[n]$")

plt.show()

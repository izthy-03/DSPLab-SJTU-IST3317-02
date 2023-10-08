"""
Exercise 2-a
"""
import numpy as np
import matplotlib.pyplot as plt

studentId = "    -No.521021911101"

fs = 5000
T = 1 / fs

f0 = 400
PI = np.pi

fig, (plt1, plt2) = plt.subplots(2, 1)


def x(t):
    return np.sin(2 * PI * f0 * t)


# Exercise a)

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

n = np.arange(0, 76, 1)
plt2.stem(n, x(n * T), label="$x[n]$", basefmt="C7-")

plt2.axis([0, 75, -1.2, 1.2])
plt2.set_xlabel("n")
plt2.set_ylabel("$x[n]$")
plt2.set_title("3 periods of $x[n]$" + studentId)
plt2.legend()

plt.show()

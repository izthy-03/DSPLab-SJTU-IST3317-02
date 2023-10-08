"""
Exercise 2-c
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

studentId = "    -No.521021911101"


def x(t):
    return np.sin(2 * PI * f0 * t)


PI = np.pi
fs = 5000
T = 1 / fs
f0 = 400

n = np.arange(0, 200, 1)
x_n = x(n * T)
# print(x_n)

fig, (plt0, plt1, plt2) = plt.subplots(3, 1)

# x[n] plot
plt0.stem(n, x_n, label="$x[n]$", basefmt="C7-")
plt0.axis([0, 59, -1.2, 1.2])
plt0.set_xlabel("$n$")
plt0.set_ylabel("$x[n]$")
plt0.set_title("$x[n]$" + studentId)
plt0.legend()

# y1[n] plot
i = np.arange(0, 76)
y1_n = 2 * x_n[2 * i + 1]

print(y1_n)
plt1.stem(i, y1_n, label="$y_1[n]$", basefmt="C7-")
plt1.axis([0, 59, -2.2, 2.2])
plt1.set_xlabel("$n$")
plt1.set_ylabel("$y_1[n]$")
plt1.set_title("$y_1[n]$" + studentId)
plt1.legend()

# y2[n] plot
i = np.arange(0, 60)
interpolator = interp1d(n, x_n, kind="linear")
y2_n = interpolator(-i / 4 + 15)
plt2.stem(i, y2_n, label="$y_2[n]$", basefmt="C7-")
plt2.axis([0, 59, -1.2, 1.2])
plt2.set_xlabel("$n$")
plt2.set_ylabel("$y_2[n]$")
plt2.set_title("$y_2[n]$ linear interpolation" + studentId)
plt2.legend()

plt.show()

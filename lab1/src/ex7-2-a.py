"""
Exercise 7-2-a
"""
import matplotlib.pyplot as plt
import numpy as np

studentId = "    -No.521021911101"
PI = np.pi

f0 = 1 / 32
A = 3.5
N = 32


def y(n):
    return A * np.cos(2 * PI * f0 * n)


def w(n):
    return np.less_equal(0, n) * np.less_equal(n, N - 1)


def DTFT_radian(freq, n, xn):
    dX = xn * np.exp(-1j * freq * n)
    return dX.sum()


def find(list, key):
    for i in range(0, len(list)):
        if list[i] >= key:
            return i


n = np.arange(-8, 40, 1)
yn = y(n)
wn = w(n)
ywn = yn * wn

plt.subplot(211)
plt.stem(n, yn, label="$y[n]$")
plt.axis([-8, 40, 0 - 4, 4])
plt.axhline(0, color="black", linewidth=2)
plt.title("$y[n]$" + studentId)
plt.xlabel("$n$")
plt.ylabel("$y[n]$")
plt.legend()
plt.grid(True)

plt.subplot(212)
plt.stem(n, ywn, label="$yw[n]$")
plt.axis([-8, 40, -4, 4])
plt.axhline(0, color="black", linewidth=2)
plt.title("$yw[n]$" + studentId)
plt.xlabel("$n$")
plt.ylabel("$yw[n]$")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

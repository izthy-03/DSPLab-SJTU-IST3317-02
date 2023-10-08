"""
Exercise 6-d
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

studentId = "    -No.521021911101"

PI = np.pi
fs = 1000
dt = 1 / fs
f1 = 16
f2 = 19
A1 = 1.4
A2 = 0.13
D = 4
df = 0.01


def x(t):
    return A1 * np.sin(2 * PI * f1 * t) + A2 * np.sin(2 * PI * f2 * t)


def CTFT(y, t, f):
    dY = y * np.exp(-1j * 2 * PI * f * t) * dt
    return dY.sum()


def CTFS(y, t, n):
    c = y * np.exp(-1j * n * 2 * PI * t) * dt
    return c.sum() / D


time = np.arange(0, D, 1 / fs)
k = np.arange(-20, 21, 1)

xt = x(time)
xk = np.array([CTFS(xt, time, i) for i in k])
print(xk)
plt.subplot(211)
plt.plot(time, xt)
plt.axis([0, D, -2, 2])
plt.title("$x(t)$  $D = 4$" + studentId)
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.grid(True)

plt.subplot(212)
plt.stem(k, np.abs(xk))
plt.axhline(0, color="black", linewidth=2)  # draw x axis
plt.axis([-20, 20, 0, xk.max() + 1])
plt.title("FS of $x(t)$ - Amplitude" + studentId)
plt.xlabel("$k$")
plt.ylabel("$|X(k)|$")
plt.grid(True)
print(np.abs(xk))

plt.tight_layout()
plt.show()

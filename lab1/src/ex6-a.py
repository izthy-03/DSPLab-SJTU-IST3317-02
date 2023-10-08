"""
Exercise 6-a
"""
import matplotlib.pyplot as plt
import numpy as np

studentId = "    -No.521021911101"

PI = np.pi
fs = 1000
dt = 1 / fs
f1 = 16
f2 = 19
A1 = 1.4
A2 = 0.13
D = 0.8
df = 0.01


def x(t):
    return A1 * np.sin(2 * PI * f1 * t) + A2 * np.sin(2 * PI * f2 * t)


def CTFT(y, t, f):
    dY = y * np.exp(-1j * 2 * PI * f * t) * dt
    return dY.sum()


# time domain
time = np.arange(0, D, 1 / fs)
xt = x(time)

plt.subplot(211)
plt.plot(time, xt)
plt.axis([0, D, -2, 2])
plt.title("$x(t)$" + studentId)
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.grid(True)

# freq domain
freq = np.arange(-30, 30, df)
xf = np.array([CTFT(xt, time, f) for f in freq])

plt.subplot(212)
plt.plot(freq, np.abs(xf))
plt.axis([-30, 30, 0, 0.6])
plt.title("FT of $x(t)$ - Amplitude" + studentId)
plt.xlabel("$f$")
plt.ylabel("$|X(f)|$")
plt.grid(True)
plt.annotate(
    "$D*A_1'/2$\n (15.998,0.564)",
    xy=(15.998, 0.564),
    xytext=(5, 0.4),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)
plt.annotate(
    "$D*A_2'/2$\n (19.038,0.100)",
    xy=(19.038, 0.100),
    xytext=(20, 0.4),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)

plt.tight_layout()
plt.show()

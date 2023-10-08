"""
Exercise 6-c
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
D = 9
df = 0.01


def x(t):
    return A1 * np.sin(2 * PI * f1 * t) + A2 * np.sin(2 * PI * f2 * t)


def CTFT(y, t, f):
    dY = y * np.exp(-1j * 2 * PI * f * t) * dt
    return dY.sum()


time = np.arange(0, D, 1 / fs)
hamming_window = signal.windows.hamming(time.size, sym=True)

xt = x(time)
xw = xt * hamming_window
# time domain
plt.subplot(211)
plt.plot(time, xw, label="$x_w(t)$")
plt.plot(time, hamming_window, label="Hamming window")

plt.title("$x_w(t)$, windowed by Hamming window" + studentId)
plt.xlabel("$t$")
plt.ylabel("$x_w(t)$")
plt.axis([0, D, -2, 2])
plt.grid(True)
plt.legend()

# freq domain
freq = np.arange(-30, 30, df)
xf = np.array([CTFT(xw, time, f) for f in freq])
hamming_FT = np.array([CTFT(hamming_window, time, f) for f in freq])
hamming_max = np.abs(hamming_FT).max()
print("Hm = ", hamming_max)

plt.subplot(212)
plt.plot(freq, np.abs(xf), label="FT of $x_w(t)$")
plt.plot(freq, np.abs(hamming_FT), label="FT of Hamming window")

plt.axis([-30, 30, 0, 7])
plt.title("FT of $x_w(t)$ and Hamming window - Amplitude" + studentId)
plt.xlabel("$f$")
plt.ylabel("$|X(f)|$")
plt.grid(True)
plt.annotate(
    "$H_m*A_1'/2$\n (16.000, 3.402)",
    xy=(16.000, 3.402),
    xytext=(10, 4.5),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)
plt.annotate(
    "$H_m*A_2'/2$\n (19.000, 0.316)",
    xy=(19.000, 0.316),
    xytext=(23, 2),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)
plt.annotate(
    "Hamming max \n$H_m=4.8595$",
    xy=(0, hamming_max),
    xytext=(1, 6),
    arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
)

plt.legend()

plt.tight_layout()
plt.show()

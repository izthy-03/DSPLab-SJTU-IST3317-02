"""
Exercise 7-1
"""
import matplotlib.pyplot as plt
import numpy as np

studentId = "    -No.521021911101"
PI = np.pi


def x(n):
    return np.less_equal(np.abs(n), 5) * (-2 * np.abs(n) + 10)


def DTFT(freq, n, xn):
    dX = xn * np.exp(-1j * 2 * PI * freq * n)
    return dX.sum()


def find(list, key):
    for i in range(0, len(list)):
        if list[i] >= key:
            return i


n = np.arange(-12, 13, 1)
xn = x(n)

df = 0.01
freq_DTFT = np.arange(-0.5, 0.5, df)
freq_DFT = np.linspace(-0.5, 0.5, n.size)
Xf_DTFT = np.array([DTFT(f, n, xn) for f in freq_DTFT])
# samples
plt.subplot(311)
plt.stem(n, xn)
plt.axis([-12, 12, 0, 12])
plt.axhline(0, color="black", linewidth=2)  # draw x axis
plt.title("$x[n]$" + studentId)
plt.xlabel("$n$")
plt.ylabel("$x[n]$")
plt.grid(True)
# DTFT
plt.subplot(312)
plt.plot(freq_DTFT, np.abs(Xf_DTFT))
plt.axis([freq_DTFT.min(), freq_DTFT.max(), 0, Xf_DTFT.max() + 5])
plt.title("DTFT of $x[n]$ - Amplitude" + studentId)
plt.xlabel("$f$")
plt.ylabel("$|X(jf)|$")
plt.grid(True)
# DFT
plt.subplot(313)

Xf_DFT = np.zeros(n.size)
for i in range(0, n.size - 1):
    k = find(freq_DTFT, freq_DFT[i])
    Xf_DFT[i] = Xf_DTFT[k]

plt.stem(freq_DFT, np.abs(Xf_DFT))
plt.axis([-0.5, 0.5, 0, Xf_DFT.max() + 5])
plt.axhline(0, color="black", linewidth=2)  # draw x axis
plt.grid(True)
plt.title("DFT of $x[n]$ - Amplitude" + studentId)
plt.xlabel("$f$")
plt.ylabel("$|X(jf)|$")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
from numpy import pi as PI
from scipy import signal as SciSig
from MyDspModule import Signal

studentID = " -No.521021911101"
MySig = Signal()

N = 2000
n = np.arange(0, N, 1)
fs = 1000
T = 1 / fs

A1 = 2
f1 = 50
A2 = 3
f2 = 55


xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)
noise = np.random.normal(0.5, 0.5, N)
xn += noise

Xjw = np.fft.fftshift(np.fft.fft(xn))
freq = np.fft.fftshift(np.fft.fftfreq(Xjw.size))

plt.subplot(311)
plt.plot(n, xn)
plt.title("Signal $y$" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")

plt.subplot(312)
plt.plot(noise, label="Normal noise")
plt.axhline(noise.sum() / noise.size, color="r", label="Average")
plt.title("Normal noise ($μ=0.5, σ=0.5$)" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(313)
plt.plot(freq * fs, np.abs(Xjw) / xn.size)
plt.xlim(-500, 500)
plt.title("Frequency spectrum-Magitude of $y$" + studentID)
plt.xlabel("Frequecy [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

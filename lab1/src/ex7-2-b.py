"""
Exercise 7-2-b
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import *

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


n = np.arange(0, N, 1)
yn = y(n)
wn = w(n)
ywn = yn * wn

df = 0.01
dtft_freq = np.arange(0, 2 * PI + df, df)
X_dtft = np.array([DTFT_radian(f, n, ywn) for f in dtft_freq])

dft_freq = np.linspace(0, 2 * PI, N)
X_dft = np.zeros(N)
for i in range(0, N - 1):
    k = find(dtft_freq, dft_freq[i])
    X_dft[i] = X_dtft[k]


plt.subplot(221)
plt.plot(dtft_freq, np.abs(X_dtft), color="orange", label="DTFT of $yw[n]$")
plt.stem(dft_freq, np.abs(X_dft), label="DFT of $yw[n]$")
plt.axhline(0, color="black", linewidth=2)
plt.grid(True)
plt.title("DFT and DTFT of $yw[n]$ - Amplitude" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(223)
plt.plot(dtft_freq, np.angle(X_dtft), color="orange", label="DTFT of $yw[n]$")
plt.stem(dft_freq, np.angle(X_dft), label="DFT of $yw[n]$")
plt.axhline(0, color="black", linewidth=2)
plt.grid(True)
plt.title("DFT and DTFT of $yw[n]$ - Phase" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("Phase")
plt.legend()

X_fft = fft(ywn, N)
fft_freq = fftfreq(N)
print(fft_freq)
fft_freq_radian = PI * np.ones(N) + 2 * PI * np.concatenate(
    (fft_freq[N // 2 :], fft_freq[: N // 2])
)
print(fft_freq_radian)

plt.subplot(222)
plt.stem(fft_freq_radian, np.abs(X_fft), label="FFT of $yw[n]$")
plt.plot(dtft_freq, np.abs(X_dtft), color="orange", label="DTFT of $yw[n]$")
plt.axhline(0, color="black", linewidth=2)
plt.grid(True)
plt.title("FFT and DTFT of $yw[n]$ - Amplitude" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("Amplitude")
plt.legend()


plt.subplot(224)
plt.stem(fft_freq_radian, np.angle(X_fft), label="FFT of $yw[n]$")
plt.plot(dtft_freq, np.angle(X_dtft), color="orange", label="DTFT of $yw[n]$")
plt.axhline(0, color="black", linewidth=2)
plt.grid(True)
plt.title("FFT and DTFT of $yw[n]$ - Phase" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("Phase")
plt.legend()

plt.show()

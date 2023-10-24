import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scpsig
from scipy import fft as scpfft
from numpy import pi as PI
from MyDspModule import Signal as mysig

studentID = " -No.521021911101"

A1 = 8
A2 = 4
f1 = 7
f2 = 24
fs = 200
T = 1 / fs

fc = 20
wc = fc / (fs / 2) * PI
beta = 3

n = np.arange(-200, 400, 1)
xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)

N = 101
fir_kaiser = scpsig.firwin(N, fc, window=("kaiser", beta), pass_zero="highpass", fs=200)

y1 = mysig.myfilter(fir_kaiser, 1, xn)

# Overlap-add method
block = []
for i in range(0, 600, 200):
    block.append(mysig.myfilter(fir_kaiser, 1, xn[i : i + 200]))

y2 = np.concatenate(([block[i][:200] for i in range(3)]))
for i in range(2):
    y2[200 * (i + 1) : 200 * (i + 1) + len(fir_kaiser)] += block[i][200:]


plt.figure(1)
plt.subplot(211)
plt.plot(n, xn, label="Original signal")
plt.plot(n, y1[: len(n)], label="Filtered signal")
plt.title("Output of myfilter() and original signal" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.axis([-200, 400, -13, 13])
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.plot(n, xn, label="Original signal")
plt.plot(n, y2, label="Filtered signal")
plt.title("Output of overlap-add method and original signal" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.axis([-200, 400, -13, 13])
plt.grid(True)
plt.legend()
plt.tight_layout()

# get the steady state of two filtered signals
y1ss = y1[len(fir_kaiser) : len(fir_kaiser) + len(xn)]
y2ss = y2[len(fir_kaiser) : len(fir_kaiser) + len(xn)]

Yjw1 = scpfft.fft(y1ss)
Yjw2 = scpfft.fft(y2ss)
freq = scpfft.fftfreq(len(y1ss), T)

plt.figure(2)
plt.plot(freq, np.abs(Yjw1))

plt.show()

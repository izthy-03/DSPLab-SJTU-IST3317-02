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
beta = 2

n = np.arange(-200, 400, 1)
xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)

N = 81
fir_kaiser = scpsig.firwin(N, fc, window=("kaiser", beta), pass_zero="highpass", fs=200)

y1 = mysig.myfilter_fir(fir_kaiser, 1, xn)

# Overlap-add method
block = []
for i in range(0, 600, 200):
    block.append(mysig.myfilter_fir(fir_kaiser, 1, xn[i : i + 200]))

y2 = np.concatenate(([block[i][:200] for i in range(3)]))
for i in range(2):
    y2[200 * (i + 1) : 200 * (i + 1) + len(fir_kaiser)] += block[i][200:]


# plt.figure(0)
# freq, Hjw = scpsig.freqz(fir_kaiser, worN=2**12)
# plt.plot(freq, np.abs(Hjw))

plt.figure(1)
plt.subplot(311)
plt.plot(n, xn, label="Original signal", linewidth=0.5)
plt.plot(n, y1[: len(n)], label="Filtered signal")
plt.title("Output of myfilter() and original signal" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.axis([-200, 400, -13, 13])
plt.grid(True)
plt.legend()

plt.subplot(312)
plt.plot(n, xn, label="Original signal", linewidth=0.5)
plt.plot(n, y2, label="Filtered signal", color="g")
plt.title("Output of overlap-add method and original signal" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.axis([-200, 400, -13, 13])
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(313)
plt.plot(n, y1[: n.size] - y2, label="Difference")
plt.title("Difference between $y_1$ and $y_2$" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.axis([-200, 400, -13, 13])
plt.grid(True)
plt.legend()
plt.tight_layout()


# get the steady state of two filtered signals
y1ss = np.concatenate((y1[fir_kaiser.size : xn.size], np.zeros(fir_kaiser.size)))
y2ss = np.concatenate((y2[fir_kaiser.size : xn.size], np.zeros(fir_kaiser.size)))
print("Steady state n = %d to %d" % (n.min() + fir_kaiser.size, xn.size))

freq = scpfft.fftshift(scpfft.fftfreq(y1ss.size, T))
Yjw0 = scpfft.fftshift(np.fft.fft(xn))
Yjw1 = scpfft.fftshift(np.fft.fft(y1ss))
Yjw2 = scpfft.fftshift(np.fft.fft(y2ss))

Y1_argmax = np.argmax(np.abs(Yjw1))
Y2_argmax = np.argmax(np.abs(Yjw2))


plt.figure(2)
plt.subplot(211)
plt.plot(freq, np.abs(Yjw0) / freq.size, label="Original signal", linewidth=0.5)
plt.plot(freq, np.abs(Yjw1) / freq.size, label="$y_1$ of myfilter()")
plt.text(
    freq[Y1_argmax],
    np.abs(Yjw1[Y1_argmax] / freq.size),
    (freq[Y1_argmax], np.abs(Yjw1[Y1_argmax] / freq.size)),
    ha="center",
)
plt.title("DFT of original signal and $y_1$" + studentID)
plt.xlabel("$f$")
plt.ylabel("$|X(j\omega)|$")
plt.axis([-fs / 2, fs / 2, 0, 4.5])
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.plot(freq, np.abs(Yjw0) / freq.size, label="Original signal", linewidth=0.5)
plt.plot(freq, np.abs(Yjw2) / freq.size, color="g", label="$y_2$ of overlap-add method")
plt.text(
    freq[Y2_argmax],
    np.abs(Yjw2[Y2_argmax] / freq.size),
    (freq[Y2_argmax], np.abs(Yjw1[Y2_argmax] / freq.size)),
    ha="center",
)
plt.title("DFT of original signal and $y_2$" + studentID)
plt.xlabel("$f$")
plt.ylabel("$|X(j\omega)|$")
plt.axis([-fs / 2, fs / 2, 0, 4.5])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

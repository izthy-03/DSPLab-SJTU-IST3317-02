import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
from scipy import signal as SciSig
from MyDspModule import Signal as MySig

studentID = " -No.521021911101"

A1 = 8
A2 = 4
f1 = 7
f2 = 24
fs = 200
T = 1 / fs

fc = 15
fcn = fc / fs
wc = 2 * PI * fc
wcn = fcn * 2 * PI
order = 16

n = np.arange(-200, 400, 1)
xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)

# Construct butterworth analog filter
filts = SciSig.lti(*SciSig.butter(order, wc, btype="low", analog=True))
filtz = SciSig.lti(*SciSig.bilinear(filts.num, filts.den, fs))

print("bz:\n", filtz.num)
print("az:\n", filtz.den)

wz, hz = SciSig.freqz(filtz.num, filtz.den)
ws, hs = SciSig.freqs(filts.num, filts.den, worN=fs * wz)

y1 = MySig.myfilter_iir(filtz.num, filtz.den, xn)
# y2 = SciSig.lfilter(filtz.num, filtz.den, xn)

Xjw = np.fft.fftshift(np.fft.fft(xn))
Yjw1 = np.fft.fftshift(np.fft.fft(y1))
freq = np.fft.fftshift(np.fft.fftfreq(Yjw1.size, 1 / fs))


plt.subplot(311)
plt.plot(n, xn, label="Original signal", linewidth=0.5)
plt.plot(n, y1, label="Filtered signal")
plt.axis([n.min(), n.max(), -15, 15])
plt.title("Time sequences of two signals" + studentID)
plt.xlabel("$n$")
plt.ylabel("$Magnitude$")
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.plot(freq, np.abs(Xjw) / freq.size, label="Original signal", linewidth=0.5)
plt.plot(freq, np.abs(Yjw1) / freq.size, label="Filtered signal")
plt.title("FRF-Magitude of two signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Magnitude$")
plt.axis([-fs / 2, fs / 2, 0, 4.5])
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(freq, np.angle(Xjw), label="Original signal", linewidth=0.5)
plt.plot(freq, np.angle(Yjw1), label="Filtered signal")
plt.title("FRF-Phase of two signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Phase$")
plt.axis([-fs / 2, fs / 2, -3.5, 3.5])
plt.legend()
plt.grid()


plt.tight_layout()
plt.show()

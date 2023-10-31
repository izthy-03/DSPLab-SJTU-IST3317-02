import matplotlib.pyplot as plt
import matplotlib.patches as patch
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

fc = 24
fcn = fc / fs
wc = 2 * PI * fc
wcn = fcn * 2 * PI

order = 16
n = np.arange(-200, 400, 1)
xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)

# Construct zpk filter
R = 0.983
z = np.array([np.exp(1j * wcn), np.exp(-1j * wcn)])
p = R * np.copy(z)
k = 1

filtzpk = SciSig.ZerosPolesGain(z, p, k)
# Convert to transfer function
filttf = filtzpk.to_tf()

y3 = MySig.myfilter_iir(filttf.num, filttf.den, xn)

# Construct butterworth analog filter
fc = 15
wc = 2 * PI * fc
filts = SciSig.lti(*SciSig.butter(order, wc, btype="low", analog=True))
filtz = SciSig.lti(*SciSig.bilinear(filts.num, filts.den, fs))
y1 = MySig.myfilter_iir(filtz.num, filtz.den, xn)

# Freq domain analysis
Xjw = np.fft.fftshift(np.fft.fft(xn))
Yjw1 = np.fft.fftshift(np.fft.fft(y1))
Yjw3 = np.fft.fftshift(np.fft.fft(y3))
freq = np.fft.fftshift(np.fft.fftfreq(Yjw1.size, 1 / fs))


plt.subplot(311)
plt.plot(n, xn, label="Original signal", linewidth=1)
plt.plot(n, y1, label="Filtered signal of butterworth filter", linewidth=1)
plt.plot(n, y3, label="Filtered signal of zpk filter", linewidth=1)

plt.axis([n.min(), n.max(), -15, 15])
plt.title("Time sequences of three signals" + studentID)
plt.xlabel("$n$")
plt.ylabel("$Magnitude$")
plt.legend()
plt.grid(True)

plt.subplot(312)
plt.plot(freq, np.abs(Xjw) / freq.size, label="Original signal")
plt.plot(
    freq,
    np.abs(Yjw1) / freq.size,
    label="Filtered signal of butterworth filter",
)
plt.plot(freq, np.abs(Yjw3) / freq.size, label="Filtered signal of zpk filter")
plt.title("FRF-Magitude of three signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Magnitude$")
plt.axis([-fs / 2, fs / 2, 0, 4.5])
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(freq, np.angle(Xjw), label="Original signal", linewidth=1)
plt.plot(
    freq, np.angle(Yjw1), label="Filtered signal of butterworth filter", linewidth=1
)
plt.plot(freq, np.angle(Yjw3), label="Filtered signal of zpk filter", linewidth=1)
plt.title("FRF-Phase of three signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Phase$")
plt.axis([-fs / 2, fs / 2, -3.5, 3.5])
plt.legend()
plt.grid()

plt.tight_layout(h_pad=0.1)
plt.show()

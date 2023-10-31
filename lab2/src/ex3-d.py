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

fc = 7
fcn = fc / fs
wc = 2 * PI * fc
wcn = 2 * PI * fcn

n = np.arange(-200, 400, 1)
xn = A1 * np.cos(2 * PI * f1 * n * T) + A2 * np.cos(2 * PI * f2 * n * T)

# Construct comb filter
R = 0.985
r = 0.01
boost = (1 - r) ** 2 / ((1 - R) * 2 * np.sin(wcn))
K = 1

z = np.array([r * np.exp(1j * wcn), r * np.exp(-1j * wcn)])
p = np.array([R * np.exp(1j * wcn), R * np.exp(-1j * wcn)])

filt_zpk = SciSig.ZerosPolesGain(z, p, K)
wz, hz = SciSig.freqz_zpk(z, p, K, worN=4096)
K = np.abs(hz).max()
filt_zpk_norm = SciSig.ZerosPolesGain(z, p, 1 / K)

filt_tf = filt_zpk_norm.to_tf()

y4 = MySig.myfilter_iir(filt_tf.num, filt_tf.den, xn)

# Construct butterworth analog filter
order = 16
fc = 15
wc = 2 * PI * fc
filts = SciSig.lti(*SciSig.butter(order, wc, btype="low", analog=True))
filtz = SciSig.lti(*SciSig.bilinear(filts.num, filts.den, fs))
y1 = MySig.myfilter_iir(filtz.num, filtz.den, xn)

# Freq domain analysis
Xjw = np.fft.fftshift(np.fft.fft(xn))
Yjw1 = np.fft.fftshift(np.fft.fft(y1))
Yjw4 = np.fft.fftshift(np.fft.fft(y4))
freq = np.fft.fftshift(np.fft.fftfreq(Xjw.size, 1 / fs))


plt.subplot(311)
plt.plot(n, xn, label="Original signal", linewidth=0.5)
plt.plot(n, y1, label="Filtered signal of butterworth filter", linewidth=1.2)
plt.plot(n, y4, label="Filtered signal of comb filter", linewidth=1.2)

plt.axis([n.min(), n.max(), -15, 15])
plt.title("Time sequences of three signals" + studentID)
plt.xlabel("$n$")
plt.ylabel("$Magnitude$")
plt.legend()
plt.grid(True)


plt.subplot(312)
plt.plot(freq, np.abs(Xjw) / freq.size, label="Original signal", linewidth=0.5)
plt.plot(
    freq,
    np.abs(Yjw1) / freq.size,
    label="Filtered signal of butterworth filter",
)
plt.plot(freq, np.abs(Yjw4) / freq.size, label="Filtered signal of comb filter")
plt.title("FRF-Magitude of three signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Magnitude$")
plt.axis([-fs / 2, fs / 2, 0, 4.5])
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(freq, np.angle(Xjw), label="Original signal", linewidth=0.5)
plt.plot(
    freq, np.angle(Yjw1), label="Filtered signal of butterworth filter", linewidth=1.5
)
plt.plot(freq, np.angle(Yjw4), label="Filtered signal of comb filter", linewidth=1.5)
plt.title("FRF-Phase of three signals" + studentID)
plt.xlabel("$Frequency$")
plt.ylabel("$Phase$")
plt.axis([-fs / 2, fs / 2, -3.5, 3.5])
plt.legend()
plt.grid()

plt.tight_layout(h_pad=0.1)

plt.show()

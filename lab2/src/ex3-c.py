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
filt_tf = filt_zpk.to_tf()

wz, hz = SciSig.freqz_zpk(z, p, K, worN=4096)
print(np.abs(hz).max())
hz /= np.abs(hz).max()

hz_dB = 20 * np.log10(np.abs(hz))
bandpath = []

for w, h in zip(wz, hz_dB):
    if h > -3:
        bandpath.append(w)

bandpath = np.array(bandpath)
bandwidth = (bandpath.max() - bandpath.min()) * fs / (2 * PI)
print("bandwidth =", bandwidth, "Hz")

# zpk image
unit_circle = patch.Circle((0, 0), 1, color="gray", fill=False)

plt.figure(1)
plt.gca().add_patch(unit_circle)
plt.scatter(
    np.real(z), np.imag(z), marker="o", facecolor="none", color="b", label="Zeroes"
)
plt.scatter(np.real(p), np.imag(p), marker="x", color="r", label="Poles")
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.gca().set_aspect("equal", adjustable="box")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.title("Pole-Zero map of the comb filter" + studentID)
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid(linewidth=0.5, linestyle="--")
plt.legend()


# FRF
plt.figure(2)
plt.subplot(211)
plt.plot(wz * fs / (2 * PI), 20 * np.log10(np.abs(hz).clip(1e-15)))
# plt.axis([0, 100, -32, 2])
plt.axhline(-3, color="r", linewidth=0.5)
plt.text(1, -2.5, "-3dB")
plt.title("FRF-Magnitude of the comb filter" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.grid(True)

plt.subplot(212)
plt.plot(wz * fs / (2 * PI), np.angle(hz))
plt.axis([0, 100, -3.5, 3.5])
plt.title("FRF-Phase of the comb filter" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.grid(True)

plt.tight_layout(h_pad=0.3)
plt.show()

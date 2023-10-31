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

print("bz:")
for bz in filtz.num:
    print(bz)
print("az:")
for az in filtz.den:
    print(az)

wz, hz = SciSig.freqz(filtz.num, filtz.den)
ws, hs = SciSig.freqs(filts.num, filts.den, worN=fs * wz)

# plt.semilogx(
#     wz * fs / (2 * np.pi),
#     20 * np.log10(np.abs(hz).clip(1e-15)),
#     label=r"$|H_z(e^{j \omega})|$",
# )
# plt.semilogx(
#     wz * fs / (2 * np.pi),
#     20 * np.log10(np.abs(hs).clip(1e-15)),
#     label=r"$|H(j \omega)|$",
# )
plt.subplot(211)
plt.plot(wz, np.abs(hz), label="Digital filter")
plt.plot(ws / fs, np.abs(hs), label="Analog butterworth filter")

plt.axis([0, PI, 0, 1.1])
plt.legend()
plt.title("FRF-Magnitude of digital filter and analog filter" + studentID)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(212)
plt.plot(wz, np.angle(hz), label="Digital filter")
plt.plot(ws / fs, np.angle(hs), label="Analog butterworth filter")

plt.axis([0, PI, -3.5, 3.5])
plt.legend()
plt.title("FRF-Phase of digital filter and analog filter" + studentID)
plt.xlabel("Frequency")
plt.ylabel("Phase")
plt.grid(True)


plt.tight_layout()
plt.show()

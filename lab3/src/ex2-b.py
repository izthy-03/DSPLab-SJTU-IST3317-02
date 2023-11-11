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

f, Sf = MySig.PSD3(xn, 2000, fs)

flist = [0, 50, 55]

for i in flist:
    print(i, "Hz:", Sf[int(f.size * i // (fs / 2))])

plt.plot(f, Sf)
plt.title("PSD of signal $y$ with direct method" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.show()

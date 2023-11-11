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

window_length = 2000

f1, Sf1 = MySig.PSD3(xn, window_length, fs)

f2, Sf2 = SciSig.welch(xn, fs, window="hamming", nperseg=window_length)


plt.figure(1)
plt.plot(f1, MySig.lin2dB_norm(Sf1), label="My PSD3 result")
plt.plot(f2, MySig.lin2dB_norm(Sf2), label="Welch method result in SciPy")
plt.title("PSD plots of two method" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.legend()

plt.figure(2)

winlen = [200, 400, 800, 1200, 1600, 2000]
for win in winlen:
    f, Sf = MySig.PSD3(xn, win, fs)
    plt.plot(f, MySig.lin2dB_norm(Sf), label="window length=" + str(win), linewidth=1)

plt.title("PSD plots of different window lengths" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude [dB]")
plt.legend()

plt.show()

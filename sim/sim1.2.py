import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
from numpy import pi as PI
from scipy import signal as SciSig
from MyDspModule import Signal as MySig

studentID = " -No.521021911101"


def harmonic(n, A, f, T):
    return A * np.cos(2 * PI * f * n * T)


fs = 80
N = np.arange(0, 1024, 1)

Ai = [6, 6, 3]
fi = [5, 8, 14]

pn = np.zeros(N.size)
for i in range(0, 3):
    pn += harmonic(N, Ai[i], fi[i], 1 / fs)

n1 = np.arange(0, 40)
n2 = np.arange(0, 180)
pn1 = np.copy(pn[:40])
pn2 = np.copy(pn[:180])

freq = np.linspace(0, 1 / 2, 4096)
Pjw1 = MySig.dtft(n1, pn1, freq)
Pjw2 = MySig.dtft(n2, pn2, freq)

# Pjw1 = np.fft.fftshift(np.fft.fft(pn1))
# Pjw2 = np.fft.fftshift(np.fft.fft(pn2))
# freq1 = np.fft.fftshift(np.fft.fftfreq(Pjw1.size, 1 / fs))
# freq2 = np.fft.fftshift(np.fft.fftfreq(Pjw2.size, 1 / fs))


plt.subplot(211)
plt.plot(freq * fs, np.abs(Pjw1))

plt.subplot(212)
plt.plot(freq * fs, np.abs(Pjw2))


plt.show()

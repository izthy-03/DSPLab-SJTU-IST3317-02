import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scpsig
from scipy import fft as scpfft
from numpy import pi as PI
from MyDspModule import Signal as mysig

studentID = " -No.521021911101"

fs = 200
fc = 20
wc = fc / (fs / 2) * PI
beta = 3

N = 101
fir_kaiser = scpsig.firwin(N, fc, window=("kaiser", beta), pass_zero="highpass", fs=200)

freq, Hjw = scpsig.freqz(fir_kaiser, worN=int(2**12))

plt.subplot(211)
plt.plot(
    freq, np.abs(Hjw), label="$h(n)$ from Kaiser window " + r"$\beta=$" + str(beta)
)
plt.title("FRF-Magnitude of $H(j\omega)$ from Kaiser" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$|H(j\omega)|$")


argmax = np.argmax(np.abs(Hjw))

plt.axhline(
    np.abs(Hjw).max(),
    0,
    freq[argmax] / freq.max(),
    linewidth=1,
    linestyle="--",
    color="black",
)
plt.axis([0, PI, 0, 1.2])
plt.grid(True)

plt.subplot(212)
plt.plot(
    freq, np.angle(Hjw), label="$h(n)$ from Kaiser window " + r"$\beta=$" + str(beta)
)
plt.title("FRF-Phase of $H(j\omega)$ from Kaiser window" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")
plt.axis([0, PI, -3.5, 3.5])
plt.grid(True)

print("Overshoot = ", np.abs(Hjw).max() - 1)

plt.tight_layout()
plt.show()

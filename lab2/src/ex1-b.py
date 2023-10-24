import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scpsig
from numpy import pi as PI
from MyDspModule import Signal as mysig

studentID = " -No.521021911101"

w_c = 0.3 * PI

N = 60
fir_kaiser = scpsig.firwin(N + 1, w_c / PI, window=("kaiser", 5), pass_zero="highpass")
fir_rec = scpsig.firwin(N + 1, w_c / PI, window="boxcar", pass_zero="highpass")
k = np.arange(0, N + 1, 1)

freq = np.arange(0, PI, 0.0001 * PI)
# freq, Hw1 = scpsig.freqz(fir_kaiser, worN=204800)
# freq, Hw2 = scpsig.freqz(fir_rec, worN=204800)

Hw1 = mysig.dtft(k, fir_kaiser, freq, form="rad")
Hw2 = mysig.dtft(k, fir_rec, freq, form="rad")

plt.figure(1)
plt.subplot(211)
plt.stem(k, fir_rec)
plt.title("$h(n)$ from rectangular window" + studentID)
plt.xlabel("$n$")
plt.ylabel("$h(n)$")
plt.axis([0, N, fir_rec.min() - 0.2, fir_rec.max() + 0.2])
plt.grid(True)

plt.subplot(212)
plt.stem(k, fir_kaiser)
plt.title("$h(n)$ from Kaiser window" + r"$(\beta=5)$" + studentID)
plt.xlabel("$n$")
plt.ylabel("$h(n)$")
plt.axis([0, N, fir_kaiser.min() - 0.2, fir_kaiser.max() + 0.2])
plt.grid(True)

plt.tight_layout()

plt.figure(2)
plt.subplot(211)
plt.plot(freq, np.abs(Hw2), label="$h(n)$ from Rectangular window")
plt.plot(freq, np.abs(Hw1), label="$h(n)$ from Kaiser window" + r"$(\beta=5)$")
plt.title("FRF-Magnitude of $H(j\omega)$ from two window types" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$|H(j\omega)|$")
plt.axis([0, PI, 0, 1.2])
plt.grid(True)
plt.legend()


plt.subplot(212)
plt.plot(freq, np.angle(Hw2), label="$h(n)$ from Rectangular window")
plt.plot(freq, np.angle(Hw1), label="$h(n)$ from Kaiser window" + r"$(\beta=5)$")
plt.title("FRF-Phase of $H(j\omega)$ from two window types" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")
plt.axis([0, PI, -3.5, 3.5])
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

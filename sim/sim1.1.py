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

pn1 = np.copy(pn[:40])
pn2 = np.copy(pn[:180])

plt.subplot(211)
plt.stem(pn1)
plt.xlim(0, pn1.size - 1)
plt.title("$p_1[n]$ plots" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(212)
plt.stem(pn2)
plt.xlim(0, pn2.size - 1)
plt.title("$p_2[n]$ plots" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()

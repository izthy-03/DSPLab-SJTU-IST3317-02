import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as PI
from MyDspModule import Signal

studentID = " -No.521021911101"

omega_c = 0.3 * PI
N1 = 30
N2 = 100


def x(rad):
    return np.less_equal(-omega_c, rad) * np.less_equal(rad, omega_c)


time = np.arange(-10, 10, 0.01)
k1 = np.arange(-N1, N1, 1)
k2 = np.arange(-N2, N2, 1)
xn = np.sin(time)
radfreq = np.arange(-PI, PI, 0.01)
xf = x(radfreq)

X_dtft = Signal.dtft(time, xn, radfreq, form="rad")
X_idtft = Signal.idtft(radfreq, xf, time, form="rad")
dk1 = Signal.idtft(radfreq, xf, k1, form="rad")
dk2 = Signal.idtft(radfreq, xf, k2, form="rad")

# h[k] of two filters
plt.figure(1)
plt.subplot(311)
plt.plot(radfreq, np.abs(xf))
plt.title("$D(\omega)$ of ideal low-pass filter" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$D(\omega)$")
plt.axis([-PI, PI, -0.1, 1.2])

plt.subplot(312)
plt.stem(k1, dk1)
plt.title("$d_1(k)$ of iDTFT of $D(\omega)$, N=60" + studentID)
plt.xlabel("$k$")
plt.ylabel("$d_1(k)$")

plt.subplot(313)
plt.stem(k2, dk2)
plt.title("$d_2(k)$ of iDTFT of $D(\omega)$, N=200" + studentID)
plt.xlabel("$k$")
plt.ylabel("$d_2(k)$")

plt.tight_layout()

plt.figure(2)

# time shift
# k1 = k1 + N1
# k2 = k2 + N2

# DTFT on h(k)
radfreq = np.arange(-PI, PI, 0.01)
Hw1 = Signal.dtft(k1, dk1, radfreq, form="rad")
Hw2 = Signal.dtft(k2, dk2, radfreq, form="rad")

# FRFs of dk1
plt.subplot(321)
plt.stem(k1, dk1)
plt.title("$d_1(k)$ of iDTFT of $D(\omega)$, N=60" + studentID)
plt.xlabel("$k$")
plt.ylabel("$d_1(k)$")

plt.subplot(323)
plt.plot(radfreq, np.abs(Hw1))
plt.grid(True)
plt.axis([-PI, PI, -0.05, 1.2])
plt.title("Magnitude of $H_1(\omega)$" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$|H_1(\omega)|$")

plt.subplot(325)
plt.plot(radfreq, np.angle(Hw1))
plt.axis([-PI, PI, -3.5, 3.5])
plt.title("Phase of $H_1(\omega)$" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")

# FRFs of dk2
plt.subplot(322)
plt.stem(k2, dk2)
plt.title("$d_2(k)$ of iDTFT of $D(\omega)$, N=200" + studentID)
plt.xlabel("$k$")
plt.ylabel("$d_2(k)$")

plt.subplot(324)
plt.plot(radfreq, np.abs(Hw2))
plt.grid(True)
plt.axis([-PI, PI, -0.05, 1.2])
plt.title("Magnitude of $H_2(\omega)$" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$|H_2(\omega)|$")

plt.subplot(326)
plt.plot(radfreq, np.angle(Hw2))
plt.axis([-PI, PI, -3.5, 3.5])
plt.title("Phase of $H_2(\omega)$" + studentID)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")


plt.tight_layout()

plt.show()

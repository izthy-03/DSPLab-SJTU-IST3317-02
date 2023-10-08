"""
Exercise 5
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

studentId = "    -No.521021911101"

PI = np.pi
D = 7
H = 3
dt = 0.001
dw = 0.05
time = np.arange(-15, 15, dt)
omega = np.arange(-10 * PI, 10 * PI, dw)

L = int(30 / dt)
N = int(20 * PI / dw)


def y(t):
    return H * np.less_equal(-D / 2, t) * np.less_equal(t, D / 2)


def CTFT(y, t, w):
    dY = y * np.exp(-1j * w * t) * dt
    return dY.sum()


# time zone
plt.subplot(311)
yt = y(time)
plt.plot(time, yt)
plt.grid(True)
plt.axis([-15, 15, 0, 5])
plt.title("$y(t)$" + studentId)
plt.xlabel("$t$")
plt.ylabel("$y(t)$")


# freq zone
yw = np.array([CTFT(yt, time, w) for w in omega])
plt.subplot(312)
# yf = fft(yt)
# xf = fftfreq(L, dt)
# plt.plot(xf, yf)
plt.plot(omega, np.abs(yw))
plt.axis([-10 * PI, 10 * PI, 0, 25])
plt.title("CTFT of $y(t)$ - Amplitude" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("$Y(\omega)$")
plt.grid(True)

plt.subplot(313)
plt.plot(omega, np.angle(yw))
plt.axis([-10 * PI, 10 * PI, -5, 5])
plt.title("CTFT of $y(t)$ - Phase" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")
plt.grid(True)

tmp = yt**2 * dt
E_time_dom = tmp.sum()
tmp = np.abs(yw) ** 2 * dw
E_freq_dom = tmp.sum() / 2 / PI

print("E in time domain = ", E_time_dom)
print("E in freq domain = ", E_freq_dom)

plt.tight_layout()
plt.show()


# y2
y2t = y(time - D / 2)

plt.subplot(311)
plt.plot(time, y2t)
plt.axis([-15, 15, 0, 5])
plt.title("$y_2(t)$" + studentId)
plt.xlabel("$t$")
plt.ylabel("$y_2(t)$")
plt.grid(True)


y2w = np.array([CTFT(y2t, time, w) for w in omega])
plt.subplot(312)
plt.plot(omega, np.abs(y2w))
plt.axis([-10 * PI, 10 * PI, 0, 25])
plt.title("CTFT of $y_2(t)$ - Amplitude" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("$Y_2(\omega)$")
plt.grid(True)

plt.subplot(313)
plt.plot(omega, np.angle(y2w))
plt.axis([-10 * PI, 10 * PI, -5, 5])
plt.title("CTFT of $y_2(t)$ - Phase" + studentId)
plt.xlabel("$\omega$")
plt.ylabel("$\phi$")
plt.grid(True)


plt.show()

"""
Exercise 4
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

studentId = "    -No.521021911101"

PI = np.pi
A = 4
f0 = 5
phi = PI / 4
T = 1 / f0
dt = 0.001
window = 2.3 * 2


def x(t):
    return A * np.sin(2 * PI * f0 * t + phi)


# time zone plot of x(t)
plt.subplot(211)
t = np.arange(-2.3, 2.3, 0.001)
x_t = x(t)
plt.plot(t, x(t))

plt.xticks(list(plt.xticks()[0]) + [-2.3, 2.3])
plt.axis([-2.3, 2.3, -5, 5])
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.title("$x(t), t\in[-2.3, 2.3]$" + studentId)


# CTFS of x(t)
def Cn(n):
    c = x_t * np.exp(-1j * n * 2 * PI * t / T)
    return c.sum() / c.size


plt.subplot(212)
k = np.arange(-5, 6, 1)
X_k = np.array([Cn(i) for i in range(-5, 6)])
print("|X(k)|=")
print(np.abs(X_k))
plt.stem(k, np.abs(X_k), basefmt="C7-")
plt.axis([-5, 5, 0, 2.3])
plt.axhline(0, color="black", linewidth=2)  # draw x axis
plt.xlabel("$k$")
plt.ylabel("$|X(k)|$")
plt.title("$|X(k)|$" + studentId)

ax = plt.gca()
x_major_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)


plt.tight_layout()
plt.show()

# CTFT of x(t)
plt.subplot(211)
df = 0.001
f_sample = np.arange(-5, 5, df)
f_window = 5 * 2


def X(f):
    dX = x_t * np.exp(-1j * 2 * PI * f * t) * dt
    return dX.sum()


CTFT = np.array([X(i) for i in f_sample])

plt.plot(f_sample, np.abs(CTFT))
plt.axis([-5, 5, 0, 10])
ax = plt.gca()
x_major_locator = MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
plt.grid(True)
plt.title("CTFT of x(t) - Amplitude" + studentId)
plt.xlabel("$f$")
plt.ylabel("$|X(f)|$")

plt.subplot(212)
plt.plot(f_sample, np.angle(CTFT))

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.grid(True)
plt.title("CTFT of x(t) - Phase" + studentId)
plt.xlabel("$f$")
plt.ylabel("$\phi$")
plt.yticks(list(plt.yticks()[0]) + [-3.14, 3.14])
plt.axis([-5, 5, -PI, PI])

E_time_domain = x_t**2 * dt
P_time_domain = E_time_domain.sum() / window

E_freq_domain = np.abs(CTFT) ** 2 * df
P_freq_domain = E_freq_domain.sum() / f_window

print("P in time domain = ", P_time_domain)
print("P in freq domain = ", P_freq_domain)

plt.tight_layout()
plt.show()

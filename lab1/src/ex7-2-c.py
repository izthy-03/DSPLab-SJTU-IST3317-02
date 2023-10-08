"""
Exercise 7-2-cd
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import *
import time

studentId = "    -No.521021911101"
PI = np.pi

f0 = 1 / 32
A = 3.5
N = 32


def y(n):
    return A * np.cos(2 * PI * f0 * n)


def w(n):
    return np.less_equal(0, n) * np.less_equal(n, N - 1)


def DTFT_radian(freq, n, xn):
    dX = xn * np.exp(-1j * freq * n)
    return dX.sum()


def find(list, key):
    for i in range(0, len(list)):
        if list[i] >= key:
            return i


n = np.arange(0, N, 1)
yn = y(n)
wn = w(n)
ywn = yn * wn


def time_dft(N, n, ywn):
    start_time = time.time()
    dft_freq = np.linspace(0, 2 * PI, N)
    X_dft = np.array([DTFT_radian(f, n, ywn) for f in dft_freq])
    end_time = time.time()
    return end_time - start_time


def time_fft(N, n, ywn):
    start_time = time.time()
    X_fft = fft(ywn, N)
    fft_freq = fftfreq(N)
    end_time = time.time()
    return end_time - start_time


tests = np.array([10, 50, 100, 500, 1000, 5000])
# tests = np.arange(10, 500000, 1000)
tests_dft = np.array([])
tests_fft = np.array([])

for N in tests:
    n = np.arange(0, N, 1)
    ywn = y(n) * w(n)
    tests_dft = np.append(tests_dft, np.array([time_dft(N, n, ywn)]))
    tests_fft = np.append(tests_fft, np.array([time_fft(N, n, ywn)]))

plt.plot(tests, tests_dft, label="time of DFT")
plt.plot(tests, tests_fft, label="time of FFT")
plt.title("Time complexity of DFT and FFT" + studentId)
plt.xlabel("$N$")
plt.ylabel("$time$ (s)")
plt.legend()

plt.show()

# d)

tests = np.array([1000000, 2**20, 2000000, 2**21])
tests_log = np.log2(tests)
tests_fft = np.array([])
for N in tests:
    n = np.arange(0, N, 1)
    ywn = y(n) * w(n)
    tests_fft = np.append(tests_fft, np.array([time_fft(N, n, ywn)]))

plt.stem(tests_log, tests_fft, label="time of FFT")
plt.title("Time complexity FFT" + studentId)
plt.xlabel("$\log_2N$")
plt.ylabel("$time$ (s)")
plt.legend()

plt.show()

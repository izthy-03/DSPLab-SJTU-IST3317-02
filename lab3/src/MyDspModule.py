import numpy as np
import math
from scipy import signal


class Signal:
    def __init__(self) -> None:
        pass

    def dtft(self, xn, freq, form="freq"):
        coe = 2 * np.pi if form == "freq" else 1
        X_dftf = np.copy(freq + 1j)
        n = np.arange(0, xn.size)
        for i in range(len(freq)):
            dX = xn * np.exp(-1j * coe * freq[i] * n)
            X_dftf[i] = dX.sum()
        return X_dftf

    def idtft(self, f, xf, time, form="freq"):
        coe = 2 * np.pi if form == "freq" else 1
        df = f[1] - f[0]
        X_idtft = np.copy(time + 1j)
        for i in range(len(time)):
            dX = xf * np.exp(1j * coe * time[i] * f) * df
            X_idtft[i] = dX.sum()
        return X_idtft / (2 * np.pi + 1 - coe)

    def myfilter_fir(self, bz, az, x):
        y = np.zeros(len(x) + len(bz))
        for i in range(len(y)):
            for k in range(len(bz)):
                y[i] += bz[k] * x[i - k] if 0 <= i - k < len(x) else 0
        return y

    def myfilter_iir(self, bz, az, x) -> np.array:
        y = np.zeros_like(x)
        m = len(bz)
        n = len(az)
        for k in range(0, len(y)):
            for i in range(0, n):
                y[k] += bz[i] * x[k - i] if k - i >= 0 else 0
            for i in range(1, m):
                y[k] -= az[i] * y[k - i] if k - i >= 0 else 0
        return y

    def mybilinear(self, b, a, fs=1):
        pass

    def autoCorrelatoin(self, xn, fs=1):
        Rn = np.zeros_like(xn)
        for tau in range(0, xn.size):
            for i in range(0, xn.size - tau):
                Rn[tau] += xn[i] * xn[i + tau]
        return Rn / xn.size

    def PSD1(self, xn, fs=1):
        slice = 500
        newsize = math.ceil(xn.size / slice) * slice
        xnpad = np.zeros(newsize)
        xnpad[: xn.size] = xn[:, 0]
        freq = np.linspace(0, 1 / 2, 8192)
        Yjw = np.zeros_like(freq, dtype=complex)
        for i in range(0, xnpad.size, slice):
            xi = xnpad[i : i + slice]
            Yjw += self.dtft(xi, freq) / (xnpad.size // slice)

        Sw = np.abs(Yjw) ** 2 / slice**2
        return freq * fs, Sw

    def PSD2(self, xn, fs=1):
        slice = 500
        newsize = math.ceil(xn.size / slice) * slice
        xnpad = np.zeros(newsize)
        xnpad[: xn.size] = xn[:, 0]
        freq = np.linspace(0, 1 / 2, 8192)
        Yjw = np.zeros_like(freq, dtype=complex)
        for i in range(0, xnpad.size, slice):
            xi = xnpad[i : i + slice]
            xcor = signal.correlate(xi, xi)
            Yjw += self.dtft(xcor, freq) / (xnpad.size // slice)

        Sw = 2 * np.abs(Yjw) / slice**2
        return freq * fs, Sw

    def PSD3(self, xn, w, fs=1):
        overlap = 0.5
        hamming = np.hamming(w)
        kaiser = np.kaiser(w, 5)
        slice = np.arange(0, xn.size - w + 1, int(w * (1 - overlap)))
        freq = np.linspace(0, 1 / 2, 8192)
        Yjw = np.zeros_like(freq, dtype=complex)
        for i in slice:
            xi = xn[i : i + w] * hamming
            Yjw += self.dtft(xi, freq) / slice.size

        Sw = 2 * np.abs(Yjw) ** 2 / np.sum(hamming) ** 2
        return freq * fs, Sw

    def lin2dB_norm(self, y):
        return 20 * np.log10(y / y.max())

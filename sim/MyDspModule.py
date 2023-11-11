import numpy as np


class Signal:
    def __init__(self) -> None:
        pass

    def dtft(n, xn, freq, form="freq"):
        coe = 2 * np.pi if form == "freq" else 1
        X_dftf = np.copy(freq + 1j)
        for i in range(freq.size):
            dX = xn * np.exp(-1j * coe * freq[i] * n)
            X_dftf[i] = dX.sum()
        return X_dftf

    def idtft(f, xf, time, form="freq"):
        coe = 2 * np.pi if form == "freq" else 1
        df = f[1] - f[0]
        X_idtft = np.copy(time + 1j)
        for i in range(len(time)):
            dX = xf * np.exp(1j * coe * time[i] * f) * df
            X_idtft[i] = dX.sum()
        return X_idtft / (2 * np.pi + 1 - coe)

    def myfilter_fir(bz, az, x):
        y = np.zeros(len(x) + len(bz))
        for i in range(len(y)):
            for k in range(len(bz)):
                y[i] += bz[k] * x[i - k] if 0 <= i - k < len(x) else 0
        return y

    def myfilter_iir(bz, az, x) -> np.array:
        y = np.zeros_like(x)
        m = len(bz)
        n = len(az)
        for k in range(0, len(y)):
            for i in range(0, n):
                y[k] += bz[i] * x[k - i] if k - i >= 0 else 0
            for i in range(1, m):
                y[k] -= az[i] * y[k - i] if k - i >= 0 else 0
        return y

    def mybilinear(b, a, fs=1):
        pass

import numpy as np


class Signal:
    def __init__(self) -> None:
        pass

    def dtft(n, xn, freq, form="freq"):
        coe = 2 * np.pi if form == "freq" else 1
        X_dftf = np.copy(freq + 1j)
        for i in range(len(freq)):
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

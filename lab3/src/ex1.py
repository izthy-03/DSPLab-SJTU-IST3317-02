import matplotlib.pyplot as plt
import matplotlib.patches as patch
import numpy as np
from numpy import pi as PI
from scipy import signal as SciSig
from MyDspModule import Signal
import pandas as pds

studentID = " -No.521021911101"
MySig = Signal()

fs = 1000

data = pds.read_excel("D:\课程作业资料\数字信号处理\lab3\data.xls")
xn = np.array(data)

f1, Sf1 = MySig.PSD1(xn, 1000)
f2, Sf2 = MySig.PSD2(xn, 1000)

print(Sf1[0])

plt.subplot(311)
plt.plot(xn)
plt.title("Original signal in data.xls-x1" + studentID)
plt.xlabel("$n$")
plt.ylabel("Magnitude")

plt.subplot(312)
plt.plot(f1, Sf1)
plt.title("psd1 with direct method(FT)" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.subplot(313)
plt.plot(f2, np.abs(Sf2))
plt.title("psd2 with indirect method(autocorrelation)" + studentID)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()

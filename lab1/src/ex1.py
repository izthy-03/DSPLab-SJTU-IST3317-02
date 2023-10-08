"""
Exercise 1
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

A = 5
B = 3.5
D = 10


def g0(t):
    return B * np.less_equal(0, t) * np.less_equal(t, A)


t = np.arange(-10, 20, 0.01)

plt.plot(t, g0(t) + g0(2 * t + D) + 2 * g0(t - D))

plt.title("CT signal $x(t)$  No.521021911101")
plt.xlabel("t")
plt.ylabel("$x(t)$")
plt.grid(linewidth=0.2)
plt.ylim(0, 8)

# 获取x轴和y轴对象
ax = plt.gca()

# 创建一个刻度间隔为2.5, 0.5的MultipleLocator对象，并将其应用于x轴和y轴上
x_major_locator = MultipleLocator(2.5)
y_major_locator = MultipleLocator(0.5)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.show()

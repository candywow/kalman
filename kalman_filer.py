# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

real = [0, 1, 2, 3, 4, 5, 4.5 , 4, 3.5, 3, 3, 3, 3, 4, 6, 8, 9, 13, 15]
measure = 2.5 * np.random.randn(1, len(real)) + real
measure = measure[0]

"""def cal_kalman(measure, p, q, r):
    x = [0, ]

    for i, val in enumerate(measure):
        p = p + q
        k = p / (p + r)
        if i == 0:
            x[i] = 0 + k * (measure[0] - 0)
        else:
            temp = x[i - 1] + k * (measure[i] - x[i - 1])
            x.append(temp)
        p = (1 - k) * p

    print x
    return x"""


f = np.array([[1, 1],
              [0, 1]])
h = np.array([[1, 0]])


def cal_kalman(measure, p, qf, t, r):
    x0 = np.array([[0], [0]])
    x = [x0, ]
    q = np.array([[t**4 * qf / 4, t**2 * qf / 2],
                  [t**2 * qf / 2, t * qf]])

    for i in range(len(measure)):
        p = np.dot(np.dot(f, p), f.T) + q
        k = np.dot(p, h.T) / (np.dot(np.dot(h, p), h.T) + r)
        if i == 0:
            temp = np.dot(k, np.array([[measure[0]]]))
            x[0] = temp
        else:
            temp = x[i - 1] + \
                np.dot(k, np.array([[measure[i]]]) - np.dot(h, x[i - 1]))
            x.append(temp)
        p = np.dot((np.eye(2) - np.dot(k, h)), p)

    return x


if __name__ == '__main__':
    p0 = np.array([[1000, 0], [0, 1000]])
    result = cal_kalman(measure, p0, 0.00001, 1, 0.1)
    x = []
    for i in range(len(result)):
        x.append(result[i][0][0])

    b, a = signal.butter(3, 0.13, 'low')
    sf = signal.filtfilt(b, a, measure)

    plt.plot(range(len(real)), x, 'bo-')
    plt.plot(range(len(real)), sf, 'yo-')
    plt.plot(range(len(real)), measure, 'gs-')
    plt.plot(range(len(real)), real, 'r^-')
    plt.show()

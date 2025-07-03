# Author: Xuewei Liu
# Email: liuxw20@mails.tsinghua.edu.cn

import numpy as np
from scipy.special import loggamma


def tweedie_reckon(y, p, mu, phi, dlambda=False):
    y0 = y == 0
    yp = y != 0

    density = np.zeros(len(y))
    density_dlambda = np.zeros(len(y))

    if type(mu) is not np.ndarray:
        mu = np.full(len(y), mu)

    if np.any(y == 0):
        lambda_pois = mu[y0] ** (2 - p) / (phi * (2 - p))
        density[y0] = np.exp(-lambda_pois)
        if dlambda:
            density_dlambda[y0] = -density[y0]

    if np.any(y != 0):
        y_sel = y[yp]
        a = (2 - p) / (1 - p)
        a1 = 1 - a
        r = -a * np.log(y_sel) + a * np.log(p - 1) - a1 * np.log(phi) - np.log(2 - p)
        # Accuracy of terms: exp(-37)
        drop = 37
        logz = max(r)
        j_max = max(y_sel ** (2 - p) / (phi * (2 - p)))
        j = max(1, j_max)
        cc = logz + a1 + a * np.log(-a)
        wmax = a1 * j_max
        estlogw = wmax
        while estlogw > (wmax - drop):
            j = j + 2
            estlogw = j * (cc - a1 * np.log(j))

        hi_j = np.ceil(j).astype(int)
        j_max = min(y_sel ** (2 - p) / (phi * (2 - p)))
        j = max(1, j_max)
        wmax = a1 * j_max
        estlogw = wmax

        while (estlogw > (wmax - drop)) and (j >= 2):
            j = max(1, j - 2)
            estlogw = j * (cc - a1 * np.log(j))

        lo_j = max(1, np.floor(j).astype(int))
        j = np.arange(lo_j, hi_j + 1)
        o = np.ones((len(y_sel), 1))
        g = np.ones((1, hi_j - lo_j + 1)) * (loggamma(j + 1) + loggamma(-a * j))
        og = o.dot(g)

        A = np.outer(r, j) - og

        m = np.max(A, axis=1)
        we = np.exp(A - m.reshape(-1, 1))
        sum_we = np.sum(we, axis=1)
        logw = np.log(sum_we) + m

        tau = phi * (p - 1) * mu[yp] ** (p - 1)
        lambda_pois = mu[yp] ** (2 - p) / (phi * (2 - p))

        logf = -y_sel / tau - lambda_pois - np.log(y_sel) + logw
        logf = np.clip(logf, -700, 700)  # 防止 underflow 或 overflow
        f = np.exp(logf)

        density[yp] = f

        if dlambda:
            A1 = A - np.log(lambda_pois).reshape(-1, 1) + np.log(j)
            m1 = np.max(A1, axis=1)
            we1 = np.exp(A1 - m1.reshape(-1, 1))
            sum_we1 = np.sum(we1, axis=1)
            logdw = np.log(sum_we1) + m1
            density_dlambda[yp] = f * (-1 + np.exp(logdw - logw))

    return density, density_dlambda

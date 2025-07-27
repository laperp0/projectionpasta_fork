"""
JAX implementation of Collignon quincuncial projection.

Author: Matthew Petroff <ORCID:0000-0002-4436-4215>
Year: 2020

This script is released into the public domain using the CC0 1.0 Public
Domain Dedication: https://creativecommons.org/publicdomain/zero/1.0/
"""

import numpy as np


def collignon_quincuncial(lambda_, phi):
    lambda_ = (lambda_ + np.pi) % (2 * np.pi)
    quadrant = (lambda_ % (2 * np.pi)) // (np.pi / 2)
    lambda_ = lambda_ % (np.pi / 2) - np.pi / 4
    cosphi = np.cos(np.abs(phi) / 2 + np.pi / 4)
    x = -2 * lambda_ * cosphi * np.sqrt(2) / np.pi
    y = (1 - 2 * (phi < 0)) * cosphi / np.sqrt(2) + (phi < 0)
    cosrot = 1 - 2 * (((quadrant + 1) % 4) // 2)
    sinrot = 1 - 2 * (quadrant // 2)
    x_t = x * cosrot - y * sinrot
    y_t = x * sinrot + y * cosrot
    return x_t, y_t


def collignon_quincuncial_inverse(x, y):
    quadrant = 0 + (y >= 0) + (x <= 0) + 2 * (y <= 0) * (x < 0)
    cosrot = 1 - 2 * (quadrant // 2)
    sinrot = 2 * (((quadrant - 1) % 4) // 2) - 1
    x_t = x * sinrot + y * cosrot
    y_t = x * cosrot - y * sinrot - 1
    y_t2 = 1 - np.abs(y_t)
    cosphi = np.sqrt(2) * y_t2 / 2
    lambda_ = np.pi / 4 * x_t / y_t2
    phi = 2 * (1 - 2 * (y_t < 0)) * (np.arccos(cosphi) - np.pi / 4)
    lambda_ += np.pi / 4 + quadrant * np.pi / 2
    return lambda_, -phi

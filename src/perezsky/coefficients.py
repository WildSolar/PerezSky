"""Coefficient tables of the Perez models.

* :data:`LUMINANCE_1993`: Perez, Seals & Michalsky (1993), "All-weather model for sky luminance
  distribution - preliminary configuration and validation", Solar Energy 50(3), 235-245.
  Shape (8 clearness categories, 5 parameters a..e, 4 coefficients x1..x4); each
  parameter is ``x1 + x2 Z + Delta (x3 + x4 Z)`` (Z: solar zenith angle in radians), except
  c and d in category 1 (see :func:`perezsky.perez_parameters`).
* :data:`EFFICACY_GLOBAL`, :data:`EFFICACY_DIFFUSE`, :data:`EFFICACY_DIRECT`: Perez, Ineichen,
  Seals, Michalsky & Stewart (1990), "Modeling daylight availability and irradiance components
  from direct and global irradiance", Solar Energy 44(5), 271-289. Shape
  (8 categories, 4 coefficients a, b, c, d).

The values are identical to those in Radiance's ``gendaylit.c``, which is the reference
implementation used for validation. Earlier MATLAB code contained three transcription errors
in the luminance table (see ``README.md``).
"""

import numpy as np

#: Lower bounds of the 8 sky clearness categories (epsilon).
EPSILON_BINS = np.array([1.0, 1.065, 1.230, 1.5, 1.95, 2.8, 4.5, 6.2])

LUMINANCE_1993 = np.array([
    [[1.3525, -0.2576, -0.2690, -1.4366], [-0.7670, 0.0007, 1.2734, -0.1233],
     [2.8000, 0.6004, 1.2375, 1.0000], [1.8734, 0.6297, 0.9738, 0.2809],
     [0.0356, -0.1246, -0.5718, 0.9938]],
    [[-1.2219, -0.7730, 1.4148, 1.1016], [-0.2054, 0.0367, -3.9128, 0.9156],
     [6.9750, 0.1774, 6.4477, -0.1239], [-1.5798, -0.5081, -1.7812, 0.1080],
     [0.2624, 0.0672, -0.2190, -0.4285]],
    [[-1.1000, -0.2515, 0.8952, 0.0156], [0.2782, -0.1812, -4.5000, 1.1766],
     [24.7219, -13.0812, -37.7000, 34.8438], [-5.0000, 1.5218, 3.9229, -2.6204],
     [-0.0156, 0.1597, 0.4199, -0.5562]],
    [[-0.5484, -0.6654, -0.2672, 0.7117], [0.7234, -0.6219, -5.6812, 2.6297],
     [33.3389, -18.3000, -62.2500, 52.0781], [-3.5000, 0.0016, 1.1477, 0.1062],
     [0.4659, -0.3296, -0.0876, -0.0329]],
    [[-0.6000, -0.3566, -2.5000, 2.3250], [0.2937, 0.0496, -5.6812, 1.8415],
     [21.0000, -4.7656, -21.5906, 7.2492], [-3.5000, -0.1554, 1.4062, 0.3988],
     [0.0032, 0.0766, -0.0656, -0.1294]],
    [[-1.0156, -0.3670, 1.0078, 1.4051], [0.2875, -0.5328, -3.8500, 3.3750],
     [14.0000, -0.9999, -7.1406, 7.5469], [-3.4000, -0.1078, -1.0750, 1.5702],
     [-0.0672, 0.4016, 0.3017, -0.4844]],
    [[-1.0000, 0.0211, 0.5025, -0.5119], [-0.3000, 0.1922, 0.7023, -1.6317],
     [19.0000, -5.0000, 1.2438, -1.9094], [-4.0000, 0.0250, 0.3844, 0.2656],
     [1.0468, -0.3788, -2.4517, 1.4656]],
    [[-1.0500, 0.0289, 0.4260, 0.3590], [-0.3250, 0.1156, 0.7781, 0.0025],
     [31.0625, -14.5000, -46.1148, 55.3750], [-7.2312, 0.4050, 13.3500, 0.6234],
     [1.5000, -0.6426, 1.8564, 0.5636]],
])

# Luminous efficacy [lm/W]: K = a + b W + c f(Z) + d g(Delta), W = precipitable water [cm].
# Global and diffuse: f = cos Z, g = ln Delta; direct: f = exp(5.73 Z - 5), g = Delta.
EFFICACY_GLOBAL = np.array([
    [96.63, -0.47, 11.50, -9.16], [107.54, 0.79, 1.79, -1.19], [98.73, 0.70, 4.40, -6.95],
    [92.72, 0.56, 8.36, -8.31], [86.73, 0.98, 7.10, -10.94], [88.34, 1.39, 6.06, -7.60],
    [78.63, 1.47, 4.93, -11.37], [99.65, 1.86, -4.46, -3.15],
])
EFFICACY_DIFFUSE = np.array([
    [97.24, -0.46, 12.00, -8.91], [107.22, 1.15, 0.59, -3.95], [104.97, 2.96, -5.53, -8.77],
    [102.39, 5.59, -13.95, -13.90], [100.71, 5.94, -22.75, -23.74], [106.42, 3.83, -36.15, -28.83],
    [141.88, 1.90, -53.24, -14.03], [152.23, 0.35, -45.27, -7.98],
])
EFFICACY_DIRECT = np.array([
    [57.20, -4.55, -2.98, 117.12], [98.99, -3.46, -1.21, 12.38], [109.83, -4.90, -1.71, -8.81],
    [110.34, -5.84, -1.99, -4.56], [106.36, -3.97, -1.75, -6.16], [107.19, -1.25, -1.51, -26.73],
    [105.75, 0.77, -1.26, -34.44], [101.18, 1.58, -1.10, -8.29],
])

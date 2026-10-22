"""Perez all-weather model: sky condition parameters, relative distribution, luminous efficacy.

All public functions take angles in degrees and accept scalars or arrays (one value per time
step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coefficients import (EFFICACY_DIFFUSE, EFFICACY_DIRECT, EFFICACY_GLOBAL, EPSILON_BINS,
                           LUMINANCE_1993)

SOLAR_CONSTANT = 1361.0          #: W/m^2 (Kopp & Lean 2011); gendaylit uses 1367
EPSILON_RANGE = (1.0, 12.01)     #: clamping range of the sky clearness (as in gendaylit)
DELTA_RANGE = (0.01, 0.6)        #: clamping range of the sky brightness (as in gendaylit)


# --- atmosphere ----------------------------------------------------------------------------

def airmass(solar_zenith, model: str = "kastenyoung1989"):
    """Relative optical air mass.

    ``model``: ``"kastenyoung1989"`` (Kasten & Young 1989, default, as pvlib) or
    ``"kasten1966"`` (Kasten 1966, as Radiance's gendaylit). Zenith angles above 90 deg are
    treated as 90 deg.
    """
    z = np.minimum(np.asarray(solar_zenith, dtype=float), 90.0)
    cz = np.cos(np.radians(z))
    if model == "kastenyoung1989":
        return 1.0 / (cz + 0.50572 * (96.07995 - z) ** -1.6364)
    if model == "kasten1966":
        return 1.0 / (cz + 0.15 * (93.885 - z) ** -1.253)
    raise ValueError(f"unknown air mass model {model!r}")


def extraterrestrial_irradiance(day_of_year, solar_constant: float = SOLAR_CONSTANT):
    """Extraterrestrial normal irradiance [W/m^2] (Earth-Sun distance after Spencer 1971).

    The annual mean is 1.00011 times ``solar_constant``.
    """
    b = 2 * np.pi * (np.asarray(day_of_year, dtype=float) - 1) / 365
    return solar_constant * (1.00011 + 0.034221 * np.cos(b) + 0.00128 * np.sin(b)
                             + 0.000719 * np.cos(2 * b) + 0.000077 * np.sin(2 * b))


# --- Perez parameters ----------------------------------------------------------------------

@dataclass
class PerezParameters:
    """Sky condition and distribution parameters for a set of time steps (arrays of length n).

    ``epsilon``/``delta`` are the sky clearness and brightness after clamping (if enabled);
    ``category`` is the clearness category 1..8; ``a``..``e`` are the parameters of the
    relative distribution (after the optional corrections, see :func:`perez_parameters`).
    """

    epsilon: np.ndarray
    delta: np.ndarray
    category: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray

    def as_array(self) -> np.ndarray:
        """Parameters a..e as an array of shape (5, n)."""
        return np.stack([self.a, self.b, self.c, self.d, self.e])

    def __len__(self):
        return self.a.size


def sky_clearness(dni, dhi, solar_zenith):
    """Perez sky clearness epsilon (infinite where DHI = 0)."""
    dni, dhi = np.asarray(dni, dtype=float), np.asarray(dhi, dtype=float)
    kz3 = 1.041 * np.radians(np.asarray(solar_zenith, dtype=float)) ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = ((dhi + dni) / dhi + kz3) / (1 + kz3)
    return np.where(dhi > 0, eps, np.inf)


def sky_brightness(dhi, solar_zenith, e0=SOLAR_CONSTANT, airmass_model: str = "kastenyoung1989"):
    """Perez sky brightness Delta = DHI * m / E0."""
    return np.asarray(dhi, dtype=float) * airmass(solar_zenith, airmass_model) / np.asarray(e0, dtype=float)


def perez_parameters(dni, dhi, solar_zenith, *, day_of_year=None, solar_constant: float = SOLAR_CONSTANT,
                     airmass_model: str = "kastenyoung1989", clamp: bool = True,
                     brightness_correction: bool = True, limit_gradation: bool = True,
                     coefficients=LUMINANCE_1993) -> PerezParameters:
    """Perez (1993) parameters for each time step.

    Parameters
    ----------
    dni, dhi : array_like
        Direct normal and diffuse horizontal irradiance [W/m^2].
    solar_zenith : array_like
        Solar zenith angle [deg].
    day_of_year : array_like, optional
        For the date-dependent extraterrestrial irradiance in the sky brightness; without it,
        ``solar_constant`` is used as is.
    airmass_model : str
        See :func:`airmass`.
    clamp : bool
        Keep the sky clearness and brightness within the range of the model,
        epsilon in [1, 12.01) and Delta in [0.01, 0.6], as Radiance's gendaylit does.
    brightness_correction : bool
        Raise Delta to at least 0.2 for 1.065 < epsilon < 2.8 before computing a..e, as
        gendaylit does (a published correction of the model for intermediate skies).
    limit_gradation : bool
        Cap the gradation parameter b at 0. For b > 0, which occurs for very bright skies
        (high Delta, e.g. cloud enhancement over snow), the gradation 1 + a exp(b / cos zeta)
        diverges towards the horizon, where the model was not fitted; vertical surfaces would
        then receive many times the global irradiance.

    With ``airmass_model="kasten1966"``, ``solar_constant=1367``, ``day_of_year`` given,
    ``limit_gradation=False`` and the other defaults, the parameters equal those of gendaylit.
    """
    dni, dhi, zen = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                          for v in (dni, dhi, solar_zenith)))
    e0 = (np.full(zen.shape, solar_constant) if day_of_year is None
          else np.broadcast_to(extraterrestrial_irradiance(day_of_year, solar_constant), zen.shape))
    dhi = np.maximum(dhi, 0.0)
    eps = sky_clearness(dni, dhi, zen)
    delta = sky_brightness(dhi, zen, e0, airmass_model)
    if clamp:
        eps = np.clip(eps, EPSILON_RANGE[0], EPSILON_RANGE[1] - 0.001)
        delta = np.clip(delta, *DELTA_RANGE)
    cat = np.clip(np.searchsorted(EPSILON_BINS, eps, side="right"), 1, 8)
    dd = delta.copy()
    if brightness_correction:
        dd = np.where((eps > 1.065) & (eps < 2.8), np.maximum(dd, 0.2), dd)
    x = coefficients[cat - 1]                         # (n, 5, 4)
    z = np.radians(zen)[:, None]
    par = x[:, :, 0] + x[:, :, 1] * z + dd[:, None] * (x[:, :, 2] + x[:, :, 3] * z)   # (n, 5)
    one = cat == 1
    if np.any(one):
        c, d = coefficients[0, 2], coefficients[0, 3]
        zz, d1 = np.radians(zen[one]), dd[one]
        with np.errstate(invalid="ignore"):
            par[one, 2] = np.exp((d1 * (c[0] + c[1] * zz)) ** c[2]) - c[3]
        par[one, 3] = -np.exp(d1 * (d[0] + d[1] * zz)) + d[2] + d1 * d[3]
    if limit_gradation:
        par[:, 1] = np.minimum(par[:, 1], 0.0)
    return PerezParameters(eps, delta, cat, *par.T.copy())


def _gradation_indicatrix(par: np.ndarray, cos_zeta, gamma, cos_gamma=None):
    """Relative distribution (1 + a exp(b / cos zeta)) (1 + c exp(d gamma) + e cos^2 gamma).

    ``par``: (5, n_t); ``cos_zeta``: (n_dirs,); ``gamma``: (n_dirs, n_t) [rad]. The gradation is
    evaluated once per distinct ``cos_zeta`` (sky grids have few distinct elevations).
    """
    a, b, c, d, e = par[:, None, :]
    cz, inverse = np.unique(np.maximum(cos_zeta, 1e-9), return_inverse=True)
    with np.errstate(over="ignore"):
        gradation = (1 + a * np.exp(b / cz[:, None]))[inverse.ravel()]
    if cos_gamma is None:
        cos_gamma = np.cos(gamma)
    return gradation * (1 + c * np.exp(d * gamma) + e * cos_gamma ** 2)


def relative_luminance(params: PerezParameters, element_zenith, angle_to_sun):
    """Relative luminance/radiance of sky elements (unnormalised Perez distribution).

    ``element_zenith`` (n_dirs,) and ``angle_to_sun`` (n_dirs, n_t) in degrees; returns
    (n_dirs, n_t). Use :func:`perezsky.perez_sky` for the distribution normalised to the
    measured irradiance.
    """
    cos_zeta = np.cos(np.radians(np.asarray(element_zenith, dtype=float))).ravel()
    gamma = np.radians(np.asarray(angle_to_sun, dtype=float)).reshape(cos_zeta.size, -1)
    return _gradation_indicatrix(params.as_array(), cos_zeta, gamma)


# --- luminous efficacy ---------------------------------------------------------------------

def luminous_efficacy(params: PerezParameters, solar_zenith, precipitable_water=2.0):
    """Perez (1990) luminous efficacy [lm/W] of the global, diffuse and direct irradiance.

    Uses the (clamped) sky clearness and brightness of ``params`` and the precipitable water
    [cm] (gendaylit's default is 2 cm). Returns ``(K_global, K_diffuse, K_direct)``; negative
    direct efficacies are set to 0, as in gendaylit.
    """
    i = params.category - 1
    z = np.radians(np.asarray(solar_zenith, dtype=float))
    w = np.asarray(precipitable_water, dtype=float)
    ln_d = np.log(params.delta)

    def model(tab, f, g):
        k = tab[i]
        return k[:, 0] + k[:, 1] * w + k[:, 2] * f + k[:, 3] * g

    k_glob = model(EFFICACY_GLOBAL, np.cos(z), ln_d)
    k_diff = model(EFFICACY_DIFFUSE, np.cos(z), ln_d)
    k_dir = np.maximum(model(EFFICACY_DIRECT, np.exp(5.73 * z - 5), params.delta), 0.0)
    return k_glob, k_diff, k_dir

"""Perez sky radiance (or luminance) on a sky grid and irradiance on planes.

The relative Perez distribution is normalised so that the sky above the horizontal yields
exactly the diffuse horizontal irradiance, and the beam (point sun or Buie circumsolar cone)
exactly DNI cos(Z). Both closures hold to machine precision for any grid resolution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .grid import SkyGrid, direction_vector, make_sky_grid, surface_normal
from .horizon import Horizon
from .model import PerezParameters, _gradation_indicatrix, luminous_efficacy, perez_parameters


def buie_sunshape(csr: float = 0.3, n_in: int = 100, n_out: int = 100):
    """Buie et al. (2003) sunshape: angles from the sun centre [mrad] and the radiance profile
    normalised to unit normal irradiance [1/sr]."""
    sunradius = 4.65
    phi_in = np.linspace(0, sunradius, n_in)
    dist_in = np.cos(0.326 * phi_in) / np.cos(0.308 * phi_in)
    phi_out = np.logspace(np.log10(sunradius), np.log10(np.pi * 1e3), n_out + 1)[1:]
    gamma = 2.2 * np.log(0.52 * csr) * csr ** 0.43 - 0.1
    kappa = 0.9 * np.log(13.5 * csr) * csr ** (-0.3)
    dist = np.concatenate([dist_in, np.exp(kappa) * phi_out ** gamma])
    phi = np.concatenate([phi_in, phi_out])
    f = dist * np.sin(phi * 1e-3)
    total = np.sum(2 * np.pi * (f[:-1] + f[1:]) / 2 * (phi[1:] - phi[:-1]) * 1e-3)
    return phi, dist / total


def _normalise(x, w, target):
    """Scale the columns of x in place such that w @ x = target."""
    total = w @ x
    with np.errstate(divide="ignore", invalid="ignore"):
        x *= np.where(total > 0, target / total, 0.0)
    return x


@dataclass
class Sky:
    """Sky radiance (or luminance) for n_t time steps on a sky grid.

    Attributes
    ----------
    grid : SkyGrid
    L : ndarray
        (n_cells, n_t) radiance [W/m^2/sr] (luminance [cd/m^2] for ``quantity="luminance"``)
        of the sky cells: Perez diffuse sky, plus the circumsolar beam for ``sun_model="buie"``,
        and the terrain radiance in cells below the horizon.
    beam : ndarray
        (n_t,) normal irradiance [W/m^2] (illuminance [lux]) of the point sun: DNI for
        ``sun_model="point"`` while the sun is visible, else 0 (the beam is then in ``L``).
    sun_elevation, sun_azimuth : ndarray
        (n_t,) [deg], azimuth from north, clockwise.
    dni, dhi : ndarray
        (n_t,) free-horizon inputs in the units of ``quantity`` (illuminances for luminance).
    params : PerezParameters
    quantity : {"radiance", "luminance"}
    """

    grid: SkyGrid
    L: np.ndarray
    beam: np.ndarray
    sun_elevation: np.ndarray
    sun_azimuth: np.ndarray
    dni: np.ndarray
    dhi: np.ndarray
    params: PerezParameters
    quantity: str = "radiance"
    sun_model: str = "point"
    horizon: Horizon | None = None

    @property
    def sun_vector(self) -> np.ndarray:
        return direction_vector(self.sun_elevation, self.sun_azimuth)

    @property
    def ghi(self) -> np.ndarray:
        """Global horizontal irradiance at the observer (behind the horizon, if any)."""
        return self.grid.horizontal_weight @ self.L + self.beam * np.maximum(self.sun_vector[:, 2], 0.0)

    def plane_irradiance(self, tilt, azimuth, albedo=0.2):
        """Irradiance on planes (tilt, azimuth [deg]) with the ground below the horizontal
        as an isotropic Lambertian reflector of the local global horizontal irradiance.

        Arguments broadcast to n_p planes. Returns a dict of (n_t, n_p) arrays ``"beam"``,
        ``"sky"`` (sky cells, incl. terrain above the horizontal and, for ``"buie"``, the
        circumsolar beam), ``"ground"`` and ``"total"``. ``albedo`` is a scalar or per time step.
        """
        normals = np.atleast_2d(surface_normal(tilt, azimuth))           # (n_p, 3)
        w = self.grid.solid_angle[:, None] * np.maximum(self.grid.vec @ normals.T, 0.0)   # (n_c, n_p)
        sky = self.L.T @ w                                                # (n_t, n_p)
        beam = self.beam[:, None] * np.maximum(self.sun_vector @ normals.T, 0.0)
        ground_view = (1 - normals[:, 2]) / 2                             # view factor to the ground
        albedo = np.broadcast_to(np.asarray(albedo, dtype=float), self.beam.shape)
        ground = (albedo * self.ghi)[:, None] * ground_view[None, :]
        return {"beam": beam, "sky": sky, "ground": ground, "total": beam + sky + ground}


def perez_sky(dni, solar_zenith, solar_azimuth, *, dhi=None, ghi=None, grid="medium",
              day_of_year=None, sun_model: str = "point", csr: float = 0.3,
              circumsolar_half_angle: float = 6.0, horizon: Horizon | None = None,
              quantity: str = "radiance", precipitable_water=2.0, **parameter_options) -> Sky:
    """Perez (1993) sky for time steps with the sun above the horizontal.

    Parameters
    ----------
    dni : array_like
        Direct normal irradiance [W/m^2], free-horizon value.
    solar_zenith, solar_azimuth : array_like
        Sun position [deg]; azimuth from north, clockwise.
    dhi, ghi : array_like
        Diffuse or global horizontal irradiance [W/m^2] (give one of them). DHI is computed
        as GHI - DNI cos(Z) and set to 0 where negative (with a warning).
    grid : str, tuple or SkyGrid
        Sky discretisation (see :func:`perezsky.make_sky_grid`).
    day_of_year : array_like, optional
        For the date-dependent extraterrestrial irradiance in the sky brightness.
    sun_model : {"point", "buie"}
        ``"point"``: the beam is returned separately as ``Sky.beam`` (exact direction).
        ``"buie"``: the beam is spread over a circumsolar cone of ``circumsolar_half_angle``
        [deg] following the Buie sunshape with circumsolar ratio ``csr``; the Perez radiance
        inside the cone is replaced by it.
    horizon : Horizon, optional
        Local horizon; cells below it get the terrain radiance, and the beam is blocked while
        the sun is behind it. ``dni``/``dhi`` are treated as free-horizon values.
    quantity : {"radiance", "luminance"}
        Luminance [cd/m^2] and illuminance [lux] from the Perez (1990) luminous efficacies of
        the direct and diffuse components (``precipitable_water`` [cm]).
    **parameter_options
        Passed to :func:`perezsky.perez_parameters` (``solar_constant``, ``airmass_model``,
        ``clamp``, ``brightness_correction``, ``limit_gradation``).
    """
    if (dhi is None) == (ghi is None):
        raise ValueError("give exactly one of dhi and ghi")
    if not isinstance(grid, SkyGrid):
        grid = make_sky_grid(grid)
    dni, zen, saz = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                          for v in (dni, solar_zenith, solar_azimuth)))
    cos_z = np.cos(np.radians(zen))
    if np.any(zen >= 90):
        raise ValueError("perez_sky needs the sun above the horizontal (solar_zenith < 90)")
    if dhi is None:
        dhi = np.asarray(ghi, dtype=float) - dni * cos_z
        if np.any(dhi < -1.0):
            warnings.warn(f"{int(np.sum(dhi < -1.0))} time step(s) with DNI cos(Z) > GHI; DHI set to 0",
                          stacklevel=2)
    dhi = np.maximum(np.broadcast_to(np.asarray(dhi, dtype=float), dni.shape), 0.0)
    params = perez_parameters(dni, dhi, zen, day_of_year=day_of_year, **parameter_options)

    if quantity == "luminance":
        _, k_diff, k_dir = luminous_efficacy(params, zen, precipitable_water)
        dni, dhi = dni * k_dir, dhi * k_diff
    elif quantity != "radiance":
        raise ValueError(f"unknown quantity {quantity!r}")

    sel = 90.0 - zen
    sun = direction_vector(sel, saz)                                   # (n_t, 3)
    cos_g = np.clip(grid.vec @ sun.T, -1.0, 1.0)                        # (n_c, n_t)
    gamma = np.arccos(cos_g)
    lv = np.maximum(_gradation_indicatrix(params.as_array(), grid.vec[:, 2], gamma, cos_g), 0.0)
    w = grid.horizontal_weight
    if sun_model == "buie":
        cell, step = np.nonzero(gamma <= np.radians(circumsolar_half_angle))
        lv[cell, step] = 0.0
    elif sun_model != "point":
        raise ValueError(f"unknown sun_model {sun_model!r}")
    L = _normalise(lv, w, dhi)
    beam = dni.copy()
    if sun_model == "buie":
        phi, dist = buie_sunshape(csr)
        cs = np.zeros_like(L)
        cs[cell, step] = np.interp(gamma[cell, step], phi * 1e-3, dist)
        L += _normalise(cs, w, dni * cos_z)
        beam = np.zeros_like(dni)
    if horizon is not None:
        masked = ~horizon.is_visible(grid.elevation, grid.azimuth)
        L[masked] = horizon.terrain_radiance(grid.elevation[masked], grid.azimuth[masked],
                                             dni, dhi, sel, saz)
        beam = beam * horizon.is_visible(sel, saz)
    return Sky(grid, L, beam, sel, saz, dni, dhi, params, quantity, sun_model, horizon)


def poa_irradiance(tilt, azimuth, dni, solar_zenith, solar_azimuth, *, dhi=None, ghi=None,
                   albedo=0.2, chunk_size: int = 512, grid="low", **sky_options):
    """Plane-of-array irradiance for long time series (computed in chunks).

    Time steps with the sun at or below the horizontal (or no irradiance) give 0. Returns a
    dict of arrays (n_t, n_p) as :meth:`Sky.plane_irradiance`; the planes (``tilt``,
    ``azimuth`` [deg]) broadcast to n_p. ``sky_options`` are passed to :func:`perez_sky`.
    """
    if (dhi is None) == (ghi is None):
        raise ValueError("give exactly one of dhi and ghi")
    if not isinstance(grid, SkyGrid):
        grid = make_sky_grid(grid)
    dni, zen, saz = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                          for v in (dni, solar_zenith, solar_azimuth)))
    other = np.broadcast_to(np.asarray(dhi if ghi is None else ghi, dtype=float), dni.shape)
    albedo = np.broadcast_to(np.asarray(albedo, dtype=float), dni.shape)
    n_p = np.atleast_2d(surface_normal(tilt, azimuth)).shape[0]
    out = {k: np.zeros((dni.size, n_p)) for k in ("beam", "sky", "ground", "total")}
    day = sky_options.pop("day_of_year", None)
    if day is not None:
        day = np.broadcast_to(np.asarray(day, dtype=float), dni.shape)
    valid = np.flatnonzero((zen < 90) & (dni + other > 0))
    for start in range(0, valid.size, chunk_size):
        i = valid[start:start + chunk_size]
        kw = {"dhi" if ghi is None else "ghi": other[i]}
        if day is not None:
            kw["day_of_year"] = day[i]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sky = perez_sky(dni[i], zen[i], saz[i], grid=grid, **kw, **sky_options)
        res = sky.plane_irradiance(tilt, azimuth, albedo[i])
        for k in out:
            out[k][i] = res[k]
    return out

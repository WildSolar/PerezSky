"""Cumulative sky: irradiation sums over a time series, fast for many viewpoints and planes.

:func:`perezsky.poa_irradiance` evaluates the Perez sky hour by hour for one horizon. When the
same weather series is needed at many viewpoints (each with its own horizon) and plane
orientations, almost all of that work is shared: the free Perez sky does not depend on the
viewpoint. :class:`CumulativeSky` computes it once, summed per sky cell and period, and then
gives the irradiation of a plane at a viewpoint with a few dot products and one vectorised
pass over the time steps for what depends on the sun being visible from that viewpoint.

The results equal the sums of :func:`perezsky.poa_irradiance` with the same horizon, terrain
model and albedo. The one approximation is in the beam reaching the terrain seen below the
horizon (``terrain="facing"``): the terrain normal depends on the cell azimuth, and cells are
grouped into ``azimuth_bins`` bins for it.
"""

from __future__ import annotations

import warnings

import numpy as np

from .grid import SkyGrid, direction_vector, make_sky_grid, surface_normal
from .horizon import Horizon
from .sky import perez_sky


class CumulativeSky:
    """Perez sky summed over a time series, per sky cell and period.

    Parameters
    ----------
    dni, solar_zenith, solar_azimuth : array_like
        Time series (n_t,): free-horizon DNI [W/m^2] and sun position [deg], azimuth from north,
        clockwise. Use the sun position at the middle of each step.
    dhi, ghi : array_like
        Diffuse or global horizontal irradiance [W/m^2] (give one of them).
    day_of_year : array_like, optional
        For the date-dependent extraterrestrial irradiance (as in :func:`perezsky.perez_sky`).
    periods : array_like of int, optional
        Period label per time step, 0 .. n_periods-1 (e.g. ``times.month - 1`` for monthly
        sums). Default: one period (the whole series).
    n_periods : int, optional
        Number of periods (default: max label + 1).
    step_hours : float
        Duration of a time step [h]. Results are sums of irradiance x ``step_hours``, i.e.
        Wh/m^2 for irradiances in W/m^2.
    grid : str, tuple or SkyGrid
        Sky discretisation (see :func:`perezsky.make_sky_grid`).
    azimuth_bins : int
        Azimuth bins for the beam on the terrain below the horizon (``terrain="facing"``).
    chunk_size : int
        Time steps per chunk when summing the sky.
    **sky_options
        Passed to :func:`perezsky.perez_sky` (``sun_model``, ``quantity``, parameter options).
        A horizon is given per call of :meth:`irradiance`, not here.

    Notes
    -----
    Time steps with the sun at or below the horizontal, or without irradiance, contribute 0,
    as in :func:`perezsky.poa_irradiance`. Memory: the sums hold n_cells x n_periods values, and
    the terrain-beam table azimuth_bins x n_daylight_steps per terrain tilt used.
    """

    def __init__(self, dni, solar_zenith, solar_azimuth, *, dhi=None, ghi=None, day_of_year=None,
                 periods=None, n_periods: int | None = None, step_hours: float = 1.0, grid="low",
                 azimuth_bins: int = 720, chunk_size: int = 512, **sky_options):
        if (dhi is None) == (ghi is None):
            raise ValueError("give exactly one of dhi and ghi")
        if "horizon" in sky_options:
            raise ValueError("pass the horizon to CumulativeSky.irradiance, not to the constructor")
        if not isinstance(grid, SkyGrid):
            grid = make_sky_grid(grid)
        self.grid = grid
        self.step_hours = float(step_hours)
        self.azimuth_bins = int(azimuth_bins)

        dni, zen, saz = np.broadcast_arrays(*(np.atleast_1d(np.asarray(v, dtype=float))
                                              for v in (dni, solar_zenith, solar_azimuth)))
        other = np.broadcast_to(np.asarray(dhi if ghi is None else ghi, dtype=float), dni.shape)
        if periods is None:
            periods = np.zeros(dni.shape, dtype=int)
        periods = np.broadcast_to(np.asarray(periods), dni.shape).astype(int)
        if periods.size and periods.min() < 0:
            raise ValueError("period labels must be >= 0")
        self.n_periods = int(n_periods if n_periods is not None else (periods.max() + 1 if periods.size else 1))
        if periods.size and periods.max() >= self.n_periods:
            raise ValueError("a period label is >= n_periods")
        day = None if day_of_year is None else np.broadcast_to(np.asarray(day_of_year, dtype=float), dni.shape)

        valid = np.flatnonzero((zen < 90) & (dni + other > 0))
        self.Lbar = np.zeros((grid.size, self.n_periods))
        parts = {k: [] for k in ("beam", "dni", "dhi", "sel", "saz", "period")}
        for start in range(0, valid.size, chunk_size):
            i = valid[start:start + chunk_size]
            kw = {"dhi" if ghi is None else "ghi": other[i]}
            if day is not None:
                kw["day_of_year"] = day[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sky = perez_sky(dni[i], zen[i], saz[i], grid=grid, **kw, **sky_options)
            self.Lbar += sky.L @ self._onehot(periods[i])
            # sky.dni/dhi are in the units of the sky (illuminance for quantity="luminance");
            # sky.beam is the point beam (0 for sun_model="buie", where the beam is in L)
            for k, v in (("beam", sky.beam), ("dni", sky.dni), ("dhi", sky.dhi), ("sel", sky.sun_elevation),
                         ("saz", sky.sun_azimuth), ("period", periods[i])):
                parts[k].append(v)
        cat = lambda k: np.concatenate(parts[k]) if parts[k] else np.zeros(0)
        self.beam_normal = cat("beam")                    # point beam per step
        self.dni = cat("dni")                             # free DNI (reaches the terrain)
        self.dhi = cat("dhi")
        self.sun_elevation = cat("sel")
        self.sun_azimuth = cat("saz")
        self.period = cat("period").astype(int)
        self._onehot_all = self._onehot(self.period)     # (T, P)

        self.sun = direction_vector(self.sun_elevation, self.sun_azimuth)              # (T, 3)
        self.sin_sun = np.maximum(self.sun[:, 2], 0.0)
        self.ghi_free = self.dhi + self.dni * self.sin_sun
        self._bin = np.floor((grid.azimuth % 360.0) / 360.0 * self.azimuth_bins).astype(int) % self.azimuth_bins
        self._terrain_cos_cache: dict[float, np.ndarray] = {}

    def _onehot(self, labels):
        m = np.zeros((np.size(labels), self.n_periods))
        m[np.arange(np.size(labels)), labels] = 1.0
        return m

    def _terrain_cos(self, terrain_tilt: float) -> np.ndarray:
        """cos(terrain normal, sun) >= 0 per azimuth bin and time step, for slopes facing the
        observer (azimuth of the cell + 180 deg)."""
        key = float(terrain_tilt)
        if key not in self._terrain_cos_cache:
            centres = (np.arange(self.azimuth_bins) + 0.5) * 360.0 / self.azimuth_bins
            normals = surface_normal(np.full(self.azimuth_bins, key), centres + 180.0)
            self._terrain_cos_cache[key] = np.maximum(normals @ self.sun.T, 0.0)
        return self._terrain_cos_cache[key]

    def irradiance(self, tilt: float, azimuth: float, horizon: Horizon | None = None,
                   albedo: float = 0.2) -> dict:
        """Irradiation of one plane at one viewpoint, per period.

        Parameters
        ----------
        tilt, azimuth : float
            Plane orientation [deg] (azimuth from north, clockwise).
        horizon : Horizon, optional
            The viewpoint's horizon; its ``albedo``, ``terrain`` and ``terrain_tilt`` define the
            radiance of the terrain seen below it (as in :func:`perezsky.perez_sky`).
        albedo : float
            Albedo of the ground below the plane (as in :meth:`Sky.plane_irradiance`).

        Returns
        -------
        dict of (n_periods,) arrays ``"beam"``, ``"sky"``, ``"ground"``, ``"total"``: sums of
        irradiance x ``step_hours`` (Wh/m^2 for hourly W/m^2 input).
        """
        g = self.grid
        normal = surface_normal(float(tilt), float(azimuth))
        weights = np.stack([g.weight(normal), g.horizontal_weight])                  # plane, horizontal
        if horizon is None:
            masked = np.zeros(g.size, dtype=bool)
            sun_visible = np.ones(self.sun_elevation.size, dtype=bool)
        else:
            masked = ~horizon.is_visible(g.elevation, g.azimuth)
            sun_visible = horizon.is_visible(self.sun_elevation, self.sun_azimuth)

        received = (weights * ~masked) @ self.Lbar                                     # (2, P)
        if masked.any():
            wm = weights[:, masked]
            a = horizon.albedo
            if horizon.terrain == "horizontal":
                received += wm.sum(axis=1)[:, None] * (a * self.ghi_free / np.pi @ self._onehot_all)[None, :]
            else:
                bt = np.radians(horizon.terrain_tilt)
                diffuse = self.dhi * (1 + np.cos(bt)) / 2 + a * self.ghi_free * (1 - np.cos(bt)) / 2
                received += wm.sum(axis=1)[:, None] * (a * diffuse / np.pi @ self._onehot_all)[None, :]
                bins = self._bin[masked]
                per_bin = np.stack([np.bincount(bins, weights=wm[k], minlength=self.azimuth_bins)
                                    for k in range(2)])                                 # (2, bins)
                beam_t = (per_bin @ self._terrain_cos(horizon.terrain_tilt)) * (self.dni * sun_visible)
                received += (a / np.pi) * (beam_t @ self._onehot_all)

        point_beam = self.beam_normal * sun_visible
        beam = (point_beam * np.maximum(self.sun @ normal, 0.0)) @ self._onehot_all
        ghi_local = received[1] + (point_beam * self.sin_sun) @ self._onehot_all
        ground = albedo * ghi_local * (1 - normal[2]) / 2
        out = {"beam": beam, "sky": received[0], "ground": ground}
        out = {k: v * self.step_hours for k, v in out.items()}
        out["total"] = out["beam"] + out["sky"] + out["ground"]
        return out

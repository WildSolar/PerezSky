"""Local horizon: masking of sky and sun, and radiance of the terrain seen instead of the sky.

A horizon is given as elevation per azimuth, as produced by horizon tools (e.g. the
``horizontool`` package or PVsyst ``.hor`` files). Sky cells below the horizon receive the
radiance of the terrain; the beam is blocked while the sun is behind the horizon.

The terrain radiance is modelled as Lambertian with albedo ``albedo`` and one of two
irradiation models:

* ``"facing"`` (default): in each direction, a terrain slope of tilt ``terrain_tilt`` (default
  30 deg) facing the observer, irradiated by the beam (if the sun is visible from the
  observer), the isotropic diffuse sky and the reflection of its own surroundings. Terrain that
  is visible from a point necessarily faces it to some degree. Against physically irradiated
  terrain slopes ray-traced with Radiance, this model deviated by at most about 2 % on the
  irradiance of tilted planes.
* ``"horizontal"``: L = albedo * GHI / pi. It ignores that terrain faces lie in their own
  shadow when the sun is behind the horizon, and that slopes facing the observer can be much
  brighter than horizontal ground (deviations up to about 14 % in the same comparison).

Negative horizon elevations (terrain below the horizontal, e.g. on mountain tops) are treated
as 0: the sky is not extended below the horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import direction_vector, surface_normal


@dataclass(frozen=True)
class Horizon:
    """Horizon profile and terrain model.

    Parameters
    ----------
    azimuth, elevation : array_like
        Horizon elevation [deg] at these azimuths [deg].
    convention : {"north", "south"}
        Azimuth convention of the input: ``"north"`` (0 = north, clockwise, as pvlib and the
        horizontool API) or ``"south"`` (0 = south, positive towards west, as PVsyst ``.hor``
        files). Stored internally from north.
    albedo : float
        Albedo of the terrain.
    terrain : {"facing", "horizontal"}
        Irradiation model of the terrain (see module docstring).
    terrain_tilt : float
        Tilt of the terrain slopes for ``terrain="facing"`` [deg].
    """

    azimuth: np.ndarray
    elevation: np.ndarray
    convention: str = "north"
    albedo: float = 0.2
    terrain: str = "facing"
    terrain_tilt: float = 30.0
    _az: np.ndarray = field(init=False, repr=False)
    _el: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        az = np.asarray(self.azimuth, dtype=float).ravel()
        el = np.asarray(self.elevation, dtype=float).ravel()
        if az.shape != el.shape or az.size < 2:
            raise ValueError("azimuth and elevation must be 1-D arrays of equal length (>= 2)")
        if self.convention == "south":
            az = az + 180.0
        elif self.convention != "north":
            raise ValueError("convention must be 'north' or 'south'")
        if self.terrain not in ("facing", "horizontal"):
            raise ValueError(f"unknown terrain model {self.terrain!r}")
        az = az % 360.0
        order = np.argsort(az)
        object.__setattr__(self, "_az", az[order])
        object.__setattr__(self, "_el", np.maximum(el[order], 0.0))

    @classmethod
    def from_file(cls, path, convention: str = "south", **kwargs) -> "Horizon":
        """Read a two-column text file (azimuth, elevation [deg]), e.g. a PVsyst ``.hor`` file.

        Lines starting with ``#`` or ``%`` are ignored. ``.hor`` files use ``convention="south"``.
        """
        data = np.loadtxt(Path(path), comments=("#", "%"), delimiter=None, ndmin=2)
        return cls(data[:, 0], data[:, 1], convention=convention, **kwargs)

    @classmethod
    def from_horizontool(cls, hor, **kwargs) -> "Horizon":
        """From a ``horizontool`` horizon object (uses its ``azimuth``/``elevation``, north = 0)."""
        return cls(np.asarray(hor.azimuth), np.asarray(hor.elevation), convention="north", **kwargs)

    def elevation_at(self, azimuth):
        """Horizon elevation [deg] at azimuths [deg, north = 0] (periodic linear interpolation)."""
        return np.interp(np.asarray(azimuth, dtype=float) % 360.0, self._az, self._el, period=360.0)

    def is_visible(self, elevation, azimuth):
        """True where a direction (e.g. the sun) lies above the horizon."""
        return np.asarray(elevation, dtype=float) >= self.elevation_at(azimuth)

    def terrain_radiance(self, elevation, azimuth, dni, dhi, sun_elevation, sun_azimuth):
        """Radiance of the terrain in the given directions [deg], shape (n_dirs, n_t).

        ``dni``, ``dhi`` and the sun position are per time step (n_t,), free-horizon values;
        the beam reaches the terrain only while the sun is visible from the observer.
        """
        dni, dhi, sel, saz = (np.atleast_1d(np.asarray(v, dtype=float))
                              for v in (dni, dhi, sun_elevation, sun_azimuth))
        sin_s = np.sin(np.radians(sel))
        ghi = dhi + dni * np.maximum(sin_s, 0.0)
        n = np.size(elevation)
        if self.terrain == "horizontal":
            return np.broadcast_to(self.albedo * ghi / np.pi, (n, ghi.size)).copy()
        bt = np.radians(self.terrain_tilt)
        normals = surface_normal(np.full(n, self.terrain_tilt), np.asarray(azimuth, dtype=float).ravel() + 180.0)
        sun = direction_vector(sel, saz)                                      # (n_t, 3)
        visible = self.is_visible(sel, saz)
        beam = (dni * visible)[None, :] * np.maximum(normals @ sun.T, 0.0)
        diffuse = dhi * (1 + np.cos(bt)) / 2 + self.albedo * ghi * (1 - np.cos(bt)) / 2
        return self.albedo * (beam + diffuse[None, :]) / np.pi

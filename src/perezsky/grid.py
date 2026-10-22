"""Discretised sky hemisphere and direction conventions.

Directions: x = east, y = north, z = up (the frame Radiance uses as well). Azimuths are
measured from north, clockwise (east = 90 deg), elevations from the horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Number of azimuthal cells in the lowest band for the named resolutions (0.5 deg bands).
RESOLUTIONS = {"low": 90, "medium": 180, "high": 360, "ultra": 720}


def direction_vector(elevation, azimuth):
    """Unit vectors (..., 3) towards directions given by elevation and azimuth [deg]."""
    el, az = np.radians(np.asarray(elevation, dtype=float)), np.radians(np.asarray(azimuth, dtype=float))
    ce = np.cos(el)
    return np.stack([np.sin(az) * ce, np.cos(az) * ce, np.sin(el)], axis=-1)


def surface_normal(tilt, azimuth):
    """Unit normal (..., 3) of a plane with ``tilt`` [deg] from horizontal, facing ``azimuth`` [deg]."""
    t, az = np.radians(np.asarray(tilt, dtype=float)), np.radians(np.asarray(azimuth, dtype=float))
    return np.stack(np.broadcast_arrays(np.sin(az) * np.sin(t), np.cos(az) * np.sin(t), np.cos(t)), axis=-1)


@dataclass(frozen=True)
class SkyGrid:
    """A discretised sky hemisphere (cells of near-constant size, exact solid angles).

    Attributes
    ----------
    elevation, azimuth : ndarray
        Cell-centre directions [deg] (azimuth from north, clockwise).
    solid_angle : ndarray
        Solid angle of each cell [sr]; they sum to 2 pi.
    vec : ndarray
        (n, 3) unit vectors towards the cell centres.
    """

    elevation: np.ndarray
    azimuth: np.ndarray
    solid_angle: np.ndarray
    vec: np.ndarray

    @property
    def size(self) -> int:
        return self.elevation.size

    @property
    def horizontal_weight(self) -> np.ndarray:
        """Solid angle times cos(zenith): ``horizontal_weight @ L`` is the horizontal irradiance."""
        return self.solid_angle * self.vec[:, 2]

    def weight(self, normal) -> np.ndarray:
        """Solid angle times the (clipped) cosine to ``normal``: ``weight(n) @ L`` is the
        irradiance on a plane with normal ``n`` from the sky above the horizontal."""
        return self.solid_angle * np.maximum(self.vec @ np.asarray(normal, dtype=float), 0.0)


def make_sky_grid(resolution="medium") -> SkyGrid:
    """Sky grid of elevation bands, each divided into azimuthal cells of near-constant size.

    ``resolution``: ``"low"`` (10'401 cells), ``"medium"`` (20'716), ``"high"`` (41'335),
    ``"ultra"`` (82'594), all with 0.5 deg bands; or a tuple ``(band_width_deg,
    n_azimuth_at_horizon)``. Solid angles are exact.
    """
    if isinstance(resolution, str):
        d_el, n_h = 0.5, RESOLUTIONS[resolution]
    else:
        d_el, n_h = resolution
    edges = np.radians(np.arange(0.0, 90.0 + 1e-9, d_el))
    if not np.isclose(edges[-1], np.pi / 2):
        raise ValueError("the elevation band width must divide 90 deg")
    el, az, omega = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mid = 0.5 * (lo + hi)
        n = max(1, int(np.ceil(n_h * np.cos(mid))))
        az_edges = np.linspace(0.0, 2 * np.pi, n + 1)
        el.append(np.full(n, mid))
        az.append(0.5 * (az_edges[:-1] + az_edges[1:]))
        omega.append(np.full(n, (2 * np.pi / n) * (np.sin(hi) - np.sin(lo))))
    el, az, omega = (np.degrees(np.concatenate(el)), np.degrees(np.concatenate(az)),
                     np.concatenate(omega))
    return SkyGrid(el, az, omega, direction_vector(el, az))

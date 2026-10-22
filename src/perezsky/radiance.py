"""Export of a Perez sky to Radiance (``brightfunc`` with a ``.cal`` file, sun as ``light`` source).

Unlike gendaylit, the exported sky has no blending with the ground brightness near the horizon
(gendaylit's ``perezlum.cal`` mixes in about 45 % ground brightness at the horizon, which
changes the irradiance on vertical planes by up to 5 %). It is normalised to the DHI of the
:class:`perezsky.Sky` by integrating the continuous distribution on a fine grid, so ray-traced
irradiances agree with :meth:`perezsky.Sky.plane_irradiance` (validated to within 0.2 % on
horizontal and tilted planes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import make_sky_grid
from .model import _gradation_indicatrix

PEREZ_CAL = """{ Perez (1993) all-weather sky (perezsky), no horizon blending.
  A1: normalisation, A2..A6: Perez a..e, A7..A9: sun direction }
skybright = if(Dz, pos(A1 * (1 + A2*Exp(A3/Dz)) * (1 + A4*Exp(A5*gam) + A6*cg*cg)), 0);
pos(x) = if(x, x, 0);
cg = Dx*A7 + Dy*A8 + Dz*A9;
gam = Acos(if(cg - 1, 1, if(-1 - cg, -1, cg)));
"""

#: Solid angle of the solar disc (0.533 deg diameter) [sr].
SUN_SOLID_ANGLE = 2 * np.pi * (1 - np.cos(np.radians(0.533 / 2)))

_FINE_GRID = None


def radiance_sky(sky, index: int = 0, ground_albedo: float = 0.2, cal_name: str = "perezsky.cal"):
    """Radiance scene description of time step ``index`` of ``sky``.

    Returns ``(rad_text, cal_text)``: write ``cal_text`` to ``cal_name`` next to the scene (or
    use :func:`write_radiance_sky`). The ground below the horizontal is a uniform glow of
    ``ground_albedo * GHI / pi``. Only ``sun_model="point"`` skies without horizon can be
    exported (model terrain as geometry instead).
    """
    global _FINE_GRID
    if sky.sun_model != "point":
        raise ValueError("only skies with sun_model='point' can be exported")
    if getattr(sky, "horizon", None) is not None:
        raise ValueError("skies with a horizon cannot be exported; model the terrain as geometry")
    if _FINE_GRID is None:
        _FINE_GRID = make_sky_grid((0.25, 1440))
    g = _FINE_GRID
    par = sky.params.as_array()[:, index:index + 1]
    sun = sky.sun_vector[index]
    cos_g = np.clip(g.vec @ sun, -1, 1)[:, None]
    lv = np.maximum(_gradation_indicatrix(par, g.vec[:, 2], np.arccos(cos_g), cos_g), 0.0)[:, 0]
    total = g.horizontal_weight @ lv
    dhi = float(sky.dhi[index])
    norm = dhi / total if total > 0 else 0.0
    ghi = dhi + float(sky.beam[index]) * max(sun[2], 0.0)
    lg = ground_albedo * ghi / np.pi
    sx, sy, sz = sun
    rad = ("# Perez sky (perezsky)\n"
           f"void brightfunc skyfunc\n2 skybright {cal_name}\n0\n"
           f"9 {norm:.8e} " + " ".join(f"{p:.8f}" for p in par[:, 0]) + f" {sx:.8f} {sy:.8f} {sz:.8f}\n"
           "skyfunc glow sky_glow\n0\n0\n4 1 1 1 0\n"
           "sky_glow source sky\n0\n0\n4 0 0 1 180\n"
           f"void glow ground_glow\n0\n0\n4 {lg:.6f} {lg:.6f} {lg:.6f} 0\n"
           "ground_glow source ground\n0\n0\n4 0 0 -1 180\n")
    if sky.beam[index] > 0:
        ls = float(sky.beam[index]) / SUN_SOLID_ANGLE
        rad += (f"void light solar\n0\n0\n3 {ls:.6e} {ls:.6e} {ls:.6e}\n"
                f"solar source sun\n0\n0\n4 {sx:.8f} {sy:.8f} {sz:.8f} 0.533\n")
    return rad, PEREZ_CAL


def write_radiance_sky(sky, path, index: int = 0, ground_albedo: float = 0.2):
    """Write the sky of time step ``index`` to ``path`` (``.rad``) and the ``.cal`` file next to it."""
    path = Path(path)
    cal = path.with_suffix(".cal")
    rad, cal_text = radiance_sky(sky, index, ground_albedo, cal_name=cal.name)
    path.write_text(rad)
    cal.write_text(cal_text)
    return path, cal

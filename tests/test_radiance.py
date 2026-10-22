"""Validation against Radiance: gendaylit (parameters, luminous efficacy) and rtrace (irradiance).

Skipped if Radiance is not found (set RADIANCE_DIR or put Radiance/bin on PATH).
"""

import itertools
import re

import numpy as np
import pytest

import perezsky as ps

pytestmark = pytest.mark.radiance

WHITE_EFFICACY = 179.0                                   # Radiance WHTEFFICACY [lm/W]
SUN_OMEGA_GENDAYLIT = 2 * np.pi * (1 - np.cos(np.radians(0.2665)))

# gendaylit's settings: Kasten (1966) air mass, E0 = 1367 W/m^2, day number 0 with -ang,
# Delta >= 0.2 for intermediate skies, no cap on b
GENDAYLIT = dict(day_of_year=0, solar_constant=1367.0, airmass_model="kasten1966", limit_gradation=False)

ALTITUDES = [5.0, 15.0, 35.0, 60.0, 80.0]
IRRADIANCE = [(0.0, 100.0), (20.0, 250.0), (150.0, 150.0), (300.0, 150.0), (500.0, 120.0),
              (700.0, 100.0), (850.0, 60.0), (900.0, 30.0), (1.6, 681.0)]


def _gendaylit(radiance, alt, az_south, dni, dhi, output):
    """(epsilon, delta, A1..A10, solar radiance), or None if gendaylit prints an error sky
    (it does so for a positive Perez parameter b)."""
    out = radiance(["gendaylit", "-ang", alt, az_south, "-W", dni, dhi, "-O", output])
    hdr = re.search(r"precipitable water content : ([\d.]+) ([\d.]+)", out)
    if hdr is None:
        return None
    eps, delta = map(float, hdr.groups())
    nums = out.split("perezlum.cal")[1].split()
    vals = [float(v) for v in nums[2:12]]                  # "0", "10", then A1..A10
    sun = re.search(r"void light solar\s+0\s+0\s+3 (\S+)", out)
    return eps, delta, vals, (float(sun.group(1)) if sun else 0.0)


def test_parameters_equal_gendaylit(radiance):
    n_compared = 0
    for alt, (dni, dhi) in itertools.product(ALTITUDES, IRRADIANCE):
        res = _gendaylit(radiance, alt, -40.0, dni, dhi, 1)
        p = ps.perez_parameters(dni, dhi, 90.0 - alt, **GENDAYLIT)
        case = (alt, dni, dhi)
        if res is None:          # gendaylit refuses skies with b > 0; perezsky caps b by default
            assert p.b[0] > 0, case
            assert ps.perez_parameters(dni, dhi, 90.0 - alt, **{**GENDAYLIT, "limit_gradation": True}).b[0] == 0
            continue
        eps, delta, vals, _ = res
        n_compared += 1
        assert p.epsilon[0] == pytest.approx(eps, abs=6e-5), case
        assert p.delta[0] == pytest.approx(delta, abs=6e-5), case
        np.testing.assert_allclose(p.as_array()[:, 0], vals[2:7], atol=2e-6, rtol=1e-6, err_msg=str(case))
    assert n_compared >= 35


def test_luminous_efficacy_equals_gendaylit(radiance):
    for alt, (dni, dhi) in itertools.product(ALTITUDES, IRRADIANCE):
        r0 = _gendaylit(radiance, alt, 30.0, dni, dhi, 0)       # visible
        r1 = _gendaylit(radiance, alt, 30.0, dni, dhi, 1)       # radiometric
        if r0 is None:
            continue
        (_, _, v0, s0), (_, _, v1, s1) = r0, r1
        p = ps.perez_parameters(dni, dhi, 90.0 - alt, **GENDAYLIT)
        _, kd, kb = ps.luminous_efficacy(p, 90.0 - alt)
        case = (alt, dni, dhi)
        if dhi > 0:
            assert kd[0] == pytest.approx(v0[0] / v1[0] * WHITE_EFFICACY, rel=2e-3), case
        if s1 > 0:
            assert kb[0] == pytest.approx(s0 / s1 * WHITE_EFFICACY, rel=2e-3, abs=0.1), case


SKIES = [(850.0, 90.0, 35.0, 140.0), (400.0, 220.0, 55.0, 250.0), (0.0, 180.0, 60.0, 180.0),
         (300.0, 60.0, 84.0, 300.0)]
PLANES = [(0, 0), (30, 180), (60, 180), (90, 180), (90, 90), (90, 0), (90, 270), (45, 135)]


def test_exported_sky_irradiance_matches_rtrace(radiance, tmp_path):
    """Plane irradiance of the discrete sky equals rtrace on the exported continuous sky."""
    albedo = 0.3
    for dni, dhi, zen, azi in SKIES:
        sky = ps.perez_sky(dni, zen, azi, dhi=dhi, grid="medium")
        rad, cal = ps.radiance_sky(sky, 0, ground_albedo=albedo)
        (tmp_path / "sky.rad").write_text(rad)
        (tmp_path / "perezsky.cal").write_text(cal)
        octree = tmp_path / "sky.oct"
        octree.write_bytes(radiance(["oconv", "sky.rad"], cwd=tmp_path).encode("latin-1"))
        normals = np.array([ps.surface_normal(t, a) for t, a in PLANES])
        rays = "\n".join(f"0 0 0 {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}" for n in normals) + "\n"
        out = radiance(["rtrace", "-h", "-I", "-ab", "1", "-aa", "0", "-ad", "65536", "-as", "0",
                        "-lw", "1e-7", "sky.oct"], stdin=rays.encode(), cwd=tmp_path)
        rt = np.array(out.split(), dtype=float).reshape(-1, 3) @ np.array([0.265, 0.670, 0.065])
        ref = sky.plane_irradiance([t for t, _ in PLANES], [a for _, a in PLANES], albedo)["total"][0]
        np.testing.assert_allclose(rt, ref, rtol=5e-3, atol=0.3, err_msg=str((dni, dhi, zen, azi)))

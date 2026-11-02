"""Cumulative sky: equal to the summed hourly poa_irradiance for any horizon, plane and period."""

import numpy as np
import pytest

import perezsky as ps


def _series(n_days=24, seed=1):
    """Hourly series over n_days spread across the year at 47 N: rough sun positions and a mix
    of clear, hazy and overcast hours (enough variety to exercise every term)."""
    rng = np.random.default_rng(seed)
    days = np.linspace(5, 360, n_days).astype(int)
    hours = np.arange(24) + 0.5
    doy, hour = (a.ravel() for a in np.meshgrid(days, hours, indexing="ij"))
    lat = np.radians(47.0)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    ha = np.radians(15.0 * (hour - 12))
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    cos_az = (np.sin(decl) - np.sin(lat) * sin_el) / (np.cos(lat) * np.cos(np.radians(el)) + 1e-12)
    az = np.degrees(np.arccos(np.clip(cos_az, -1, 1)))
    az = np.where(hour > 12, 360 - az, az)
    clear = rng.uniform(0, 1, doy.size)
    up = np.maximum(sin_el, 0)
    dni = np.where(el > 0, 900 * clear * up ** 0.3, 0.0)
    dhi = np.where(el > 0, (60 + 250 * (1 - clear)) * up ** 0.8, 0.0)
    month = np.minimum((doy - 1) // 31, 11)
    return dict(dni=dni, zen=90 - el, az=az, dhi=dhi, doy=doy, month=month)


S = _series()
HORIZONS = {
    "none": None,
    "flat": ps.Horizon([0, 90, 180, 270], [0, 0, 0, 0]),
    "valley": ps.Horizon(np.arange(0, 360, 1.0), 12 + 10 * np.cos(np.radians(np.arange(0, 360, 1.0)) * 2)),
    "street": ps.Horizon(np.arange(0, 360, 1.0),
                         np.where(np.abs(((np.arange(0, 360, 1.0) - 90) % 180) - 90) < 60, 35.0, 4.0),
                         albedo=0.3),
    "horizontal": ps.Horizon(np.arange(0, 360, 10.0), np.linspace(5, 40, 36), terrain="horizontal"),
}
PLANES = [(0, 0), (30, 180), (45, 100), (90, 180), (90, 0), (60, 300)]


@pytest.fixture(scope="module")
def cum():
    return ps.CumulativeSky(S["dni"], S["zen"], S["az"], dhi=S["dhi"], day_of_year=S["doy"],
                            periods=S["month"], n_periods=12, grid="low")


@pytest.mark.parametrize("name", list(HORIZONS))
def test_matches_hourly(cum, name):
    hor = HORIZONS[name]
    for tilt, az in PLANES:
        ref = ps.poa_irradiance(tilt, az, S["dni"], S["zen"], S["az"], dhi=S["dhi"], albedo=0.25,
                                day_of_year=S["doy"], grid="low", horizon=hor)
        got = cum.irradiance(tilt, az, horizon=hor, albedo=0.25)
        for k in ("beam", "sky", "ground", "total"):
            monthly = np.bincount(S["month"], weights=ref[k][:, 0], minlength=12)
            # exact except the beam on the terrain, binned by azimuth (0.5 deg)
            tol = dict(rtol=1e-9, atol=1e-6) if name in ("none", "flat", "horizontal") else dict(rtol=2e-4, atol=0.05)
            np.testing.assert_allclose(got[k], monthly, err_msg=f"{name} {tilt}/{az} {k}", **tol)


def _hourly_with_overhang(tilt, az, hor, over, albedo):
    """Reference: hourly sky with the overhang applied cell by cell (soffit lit by the ground)."""
    import warnings
    grid = ps.make_sky_grid("low")
    covered = grid.elevation > over.elevation_at(grid.azimuth)
    valid = np.flatnonzero((S["zen"] < 90) & (S["dni"] + S["dhi"] > 0))
    out = {k: np.zeros(12) for k in ("beam", "sky", "ground", "total")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sky = ps.perez_sky(S["dni"][valid], S["zen"][valid], S["az"][valid], dhi=S["dhi"][valid],
                           day_of_year=S["doy"][valid], grid=grid, horizon=hor)
    ghi_free = S["dhi"][valid] + S["dni"][valid] * np.cos(np.radians(S["zen"][valid]))
    sky.L[covered] = over.albedo * albedo * ghi_free / np.pi
    sky.beam = sky.beam * (90.0 - S["zen"][valid] <= over.elevation_at(S["az"][valid]))
    res = sky.plane_irradiance(tilt, az, albedo)
    for k in out:
        out[k] = np.bincount(S["month"][valid], weights=res[k][:, 0], minlength=12)
    return out


@pytest.mark.parametrize("name", ["none", "street"])
def test_overhang_matches_hourly(cum, name):
    az = np.arange(0, 360, 1.0)
    phi = np.radians((az - 180 + 180) % 360 - 180)                 # eave above a south façade
    el = np.where(np.cos(phi) > 1e-6, np.degrees(np.arctan2(2.0, 0.8 / np.maximum(np.cos(phi), 1e-6))), 90.0)
    over = ps.Overhang(az, el, albedo=0.3)
    for tilt, plane_az in [(90, 180), (60, 200), (0, 0)]:
        ref = _hourly_with_overhang(tilt, plane_az, HORIZONS[name], over, 0.25)
        got = cum.irradiance(tilt, plane_az, HORIZONS[name], 0.25, overhang=over)
        for k in ref:
            np.testing.assert_allclose(got[k], ref[k], rtol=2e-4, atol=0.05, err_msg=f"{name} {tilt} {k}")
    # nothing above (90 deg everywhere) changes nothing
    none = ps.Overhang([0, 180], [90, 90])
    np.testing.assert_allclose(cum.irradiance(30, 180, HORIZONS[name], overhang=none)["total"],
                               cum.irradiance(30, 180, HORIZONS[name])["total"], rtol=1e-12)


def test_periods_and_step_hours():
    one = ps.CumulativeSky(S["dni"], S["zen"], S["az"], dhi=S["dhi"], grid="low", step_hours=0.5)
    twelve = ps.CumulativeSky(S["dni"], S["zen"], S["az"], dhi=S["dhi"], grid="low", periods=S["month"])
    hor = HORIZONS["valley"]
    a, b = one.irradiance(30, 200, hor), twelve.irradiance(30, 200, hor)
    assert a["total"].shape == (1,) and b["total"].shape == (12,)
    np.testing.assert_allclose(a["total"][0], 0.5 * b["total"].sum(), rtol=1e-10)


def test_ghi_input_and_night():
    ghi = S["dhi"] + S["dni"] * np.maximum(np.cos(np.radians(S["zen"])), 0)
    a = ps.CumulativeSky(S["dni"], S["zen"], S["az"], dhi=S["dhi"], grid="low")
    b = ps.CumulativeSky(S["dni"], S["zen"], S["az"], ghi=ghi, grid="low")
    np.testing.assert_allclose(a.irradiance(90, 180)["total"], b.irradiance(90, 180)["total"], rtol=1e-9)
    assert a.sun_elevation.min() > 0                      # night steps are skipped
    with pytest.raises(ValueError):
        ps.CumulativeSky(S["dni"], S["zen"], S["az"], grid="low")
    with pytest.raises(ValueError):
        ps.CumulativeSky(S["dni"], S["zen"], S["az"], dhi=S["dhi"], horizon=HORIZONS["flat"])

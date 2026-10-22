"""Sky distribution: closures, plane irradiance, horizon, luminance, chunked time series."""

import numpy as np
import pytest

import perezsky as ps

# clear (sun south-east), hazy (west), overcast, bright overcast, low sun (north-west)
DNI = np.array([850.0, 400.0, 0.0, 1.6, 300.0])
DHI = np.array([90.0, 220.0, 180.0, 681.0, 60.0])
ZEN = np.array([35.0, 55.0, 60.0, 48.6, 84.0])
AZI = np.array([140.0, 250.0, 180.0, 184.0, 300.0])


@pytest.mark.parametrize("sun_model", ["point", "buie"])
def test_horizontal_closure(sun_model):
    sky = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low", sun_model=sun_model)
    ghi = DHI + DNI * np.cos(np.radians(ZEN))
    np.testing.assert_allclose(sky.ghi, ghi, rtol=1e-10)
    res = sky.plane_irradiance(0.0, 0.0)
    np.testing.assert_allclose(res["total"][:, 0], ghi, rtol=1e-10)
    np.testing.assert_allclose(res["ground"], 0.0, atol=1e-12)
    if sun_model == "point":
        np.testing.assert_allclose(sky.grid.horizontal_weight @ sky.L, DHI, rtol=1e-10)


def test_dhi_or_ghi_input():
    ghi = DHI + DNI * np.cos(np.radians(ZEN))
    a = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low")
    b = ps.perez_sky(DNI, ZEN, AZI, ghi=ghi, grid="low")
    np.testing.assert_allclose(a.L, b.L, rtol=1e-9, atol=1e-12)
    with pytest.raises(ValueError):
        ps.perez_sky(DNI, ZEN, AZI, grid="low")


def test_grid_convergence_tilted_planes():
    """Plane irradiance on the 'low' grid is within 0.3 % of the 'ultra' grid."""
    tilt = np.array([30, 60, 90, 90, 90, 120])
    az = np.array([180, 180, 180, 90, 0, 270])
    lo = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low").plane_irradiance(tilt, az)["total"]
    hi = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="ultra").plane_irradiance(tilt, az)["total"]
    np.testing.assert_allclose(lo, hi, rtol=3e-3, atol=0.5)


def test_buie_and_point_agree_on_planes():
    tilt, az = np.array([30, 90, 90]), np.array([180, 180, 90])
    p = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="medium").plane_irradiance(tilt, az)["total"]
    b = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="medium", sun_model="buie").plane_irradiance(tilt, az)["total"]
    np.testing.assert_allclose(b, p, rtol=0.03, atol=2.0)


def test_isotropic_ground_and_vertical_plane():
    sky = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low")
    res = sky.plane_irradiance(90.0, 180.0, albedo=0.3)
    np.testing.assert_allclose(res["ground"][:, 0], 0.3 * sky.ghi / 2)
    beam = DNI * np.maximum(ps.direction_vector(90 - ZEN, AZI) @ ps.surface_normal(90.0, 180.0), 0)
    np.testing.assert_allclose(res["beam"][:, 0], beam)


def test_horizon():
    flat = ps.Horizon([0, 90, 180, 270], [0, 0, 0, 0])
    a = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low")
    b = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low", horizon=flat)
    np.testing.assert_allclose(a.L, b.L)
    high = ps.Horizon([0, 90, 180, 270], [70, 70, 70, 70], albedo=0.5, terrain="horizontal")
    h = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low", horizon=high)
    assert np.all(h.beam[ZEN > 20] == 0)
    below = h.grid.elevation < 69.9
    ghi = DHI + DNI * np.cos(np.radians(ZEN))
    np.testing.assert_allclose(h.L[below], np.broadcast_to(0.5 * ghi / np.pi, h.L[below].shape))
    # south-convention input (PVsyst .hor): 0 = south
    s = ps.Horizon([0, 90, 180, 270], [10, 20, 30, 40], convention="south")
    assert s.elevation_at(180.0) == pytest.approx(10.0)      # south
    assert s.elevation_at(270.0) == pytest.approx(20.0)      # west


def test_horizon_from_file(tmp_path):
    f = tmp_path / "test.hor"
    f.write_text("# az el\n-180 5\n-90 10\n0 20\n90 15\n")
    h = ps.Horizon.from_file(f)
    assert h.elevation_at(180.0) == pytest.approx(20.0)      # south
    assert h.elevation_at(0.0) == pytest.approx(5.0)         # north


def test_luminance_closure():
    sky = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low", quantity="luminance")
    _, kd, kb = ps.luminous_efficacy(sky.params, ZEN)
    ev = DHI * kd + DNI * kb * np.cos(np.radians(ZEN))
    np.testing.assert_allclose(sky.ghi, ev, rtol=1e-10)
    assert np.all((sky.ghi / (DHI + DNI * np.cos(np.radians(ZEN)))) > 80)     # lm/W


def test_poa_time_series_matches_sky():
    zen = np.concatenate([ZEN, [95.0, 92.0]])
    dni = np.concatenate([DNI, [0.0, 0.0]])
    dhi = np.concatenate([DHI, [0.0, 0.0]])
    azi = np.concatenate([AZI, [0.0, 10.0]])
    tilt, az = np.array([30, 90]), np.array([180, 270])
    out = ps.poa_irradiance(tilt, az, dni, zen, azi, dhi=dhi, albedo=0.25, chunk_size=2, grid="low")
    ref = ps.perez_sky(DNI, ZEN, AZI, dhi=DHI, grid="low").plane_irradiance(tilt, az, 0.25)
    np.testing.assert_allclose(out["total"][:5], ref["total"])
    assert np.all(out["total"][5:] == 0)


def test_plot_sky():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from perezsky.plot import plot_sky
    ax = plot_sky(ps.perez_sky(DNI[:1], ZEN[:1], AZI[:1], dhi=DHI[:1], grid="low"))
    assert ax.name == "polar"


def test_radiance_export_text():
    sky = ps.perez_sky(DNI[:1], ZEN[:1], AZI[:1], dhi=DHI[:1], grid="low")
    rad, cal = ps.radiance_sky(sky, 0)
    assert "brightfunc skyfunc" in rad and "solar source sun" in rad and "skybright" in cal
    with pytest.raises(ValueError):
        ps.radiance_sky(ps.perez_sky(DNI[:1], ZEN[:1], AZI[:1], dhi=DHI[:1], grid="low", sun_model="buie"))

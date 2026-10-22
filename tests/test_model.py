"""Perez parameters, atmosphere helpers and luminous efficacy."""

import numpy as np
import pytest

import perezsky as ps
from perezsky.coefficients import EPSILON_BINS

# Radiance gendaylit.c, coeff_perez[] in the order of the source file (independent transcription)
GENDAYLIT_COEFF = [
    1.3525, -0.2576, -0.2690, -1.4366, -0.7670, 0.0007, 1.2734, -0.1233, 2.8000, 0.6004, 1.2375, 1.000,
    1.8734, 0.6297, 0.9738, 0.2809, 0.0356, -0.1246, -0.5718, 0.9938, -1.2219, -0.7730, 1.4148, 1.1016,
    -0.2054, 0.0367, -3.9128, 0.9156, 6.9750, 0.1774, 6.4477, -0.1239, -1.5798, -0.5081, -1.7812, 0.1080,
    0.2624, 0.0672, -0.2190, -0.4285, -1.1000, -0.2515, 0.8952, 0.0156, 0.2782, -0.1812, -4.5000, 1.1766,
    24.7219, -13.0812, -37.7000, 34.8438, -5.0000, 1.5218, 3.9229, -2.6204, -0.0156, 0.1597, 0.4199,
    -0.5562, -0.5484, -0.6654, -0.2672, 0.7117, 0.7234, -0.6219, -5.6812, 2.6297, 33.3389, -18.3000,
    -62.2500, 52.0781, -3.5000, 0.0016, 1.1477, 0.1062, 0.4659, -0.3296, -0.0876, -0.0329, -0.6000,
    -0.3566, -2.5000, 2.3250, 0.2937, 0.0496, -5.6812, 1.8415, 21.0000, -4.7656, -21.5906, 7.2492,
    -3.5000, -0.1554, 1.4062, 0.3988, 0.0032, 0.0766, -0.0656, -0.1294, -1.0156, -0.3670, 1.0078, 1.4051,
    0.2875, -0.5328, -3.8500, 3.3750, 14.0000, -0.9999, -7.1406, 7.5469, -3.4000, -0.1078, -1.0750,
    1.5702, -0.0672, 0.4016, 0.3017, -0.4844, -1.0000, 0.0211, 0.5025, -0.5119, -0.3000, 0.1922, 0.7023,
    -1.6317, 19.0000, -5.0000, 1.2438, -1.9094, -4.0000, 0.0250, 0.3844, 0.2656, 1.0468, -0.3788,
    -2.4517, 1.4656, -1.0500, 0.0289, 0.4260, 0.3590, -0.3250, 0.1156, 0.7781, 0.0025, 31.0625, -14.5000,
    -46.1148, 55.3750, -7.2312, 0.4050, 13.3500, 0.6234, 1.5000, -0.6426, 1.8564, 0.5636,
]


def test_luminance_table_equals_gendaylit():
    np.testing.assert_array_equal(ps.LUMINANCE_1993.ravel(), GENDAYLIT_COEFF)


def test_efficacy_tables_spot_values():
    # gendaylit.c glob_h_effi_PEREZ / glob_h_diffuse_effi_PEREZ / direct_n_effi_PEREZ
    assert ps.EFFICACY_GLOBAL[0].tolist() == [96.63, -0.47, 11.50, -9.16]
    assert ps.EFFICACY_DIFFUSE[7].tolist() == [152.23, 0.35, -45.27, -7.98]
    assert ps.EFFICACY_DIRECT[0].tolist() == [57.20, -4.55, -2.98, 117.12]
    assert ps.EFFICACY_DIRECT[7].tolist() == [101.18, 1.58, -1.10, -8.29]


def test_airmass_and_extraterrestrial_against_pvlib():
    pvlib = pytest.importorskip("pvlib")
    z = np.linspace(0, 89.5, 50)
    for model in ("kastenyoung1989", "kasten1966"):
        np.testing.assert_allclose(ps.airmass(z, model), pvlib.atmosphere.get_relative_airmass(z, model),
                                   rtol=1e-10)
    doy = np.arange(1, 366)
    ref = pvlib.irradiance.get_extra_radiation(doy, solar_constant=1361.0, method="spencer")
    np.testing.assert_allclose(ps.extraterrestrial_irradiance(doy), ref, rtol=1e-10)
    assert ps.extraterrestrial_irradiance(doy).mean() == pytest.approx(1361.0 * 1.00011, rel=1e-6)


def test_clearness_and_categories():
    z = np.array([30.0, 30.0, 60.0])
    eps = ps.sky_clearness([0.0, 500.0, 100.0], [100.0, 100.0, 0.0], z)
    kz3 = 1.041 * np.radians(30.0) ** 3
    assert eps[0] == pytest.approx(1.0)
    assert eps[1] == pytest.approx((6 + kz3) / (1 + kz3))
    assert np.isinf(eps[2])
    # category boundaries: lower bound inclusive, as in gendaylit
    p = ps.perez_parameters(EPSILON_BINS * 0 + 0.0, np.full(8, 100.0), np.full(8, 0.0))
    assert (p.category == 1).all()
    dhi = 100.0
    for k, lo in enumerate(EPSILON_BINS):
        dni = (lo - 1) * dhi + 1e-9           # at Z = 0: epsilon = 1 + DNI / DHI
        assert ps.perez_parameters(dni, dhi, 0.0, clamp=False).category[0] == k + 1


def test_clamping_and_corrections():
    # very bright overcast sky (Totalp 2025-03-16): Delta > 0.6, b > 0 without limits
    raw = ps.perez_parameters(1.6, 681.3, 48.6, clamp=False, brightness_correction=False, limit_gradation=False)
    assert raw.delta[0] > 0.6 and raw.b[0] > 0
    p = ps.perez_parameters(1.6, 681.3, 48.6)
    assert p.delta[0] == 0.6 and p.b[0] <= 0
    # brightness correction for intermediate skies: Delta raised to 0.2 for the parameters only
    q1 = ps.perez_parameters(50.0, 30.0, 40.0, brightness_correction=True)
    q0 = ps.perez_parameters(50.0, 30.0, 40.0, brightness_correction=False)
    assert 1.065 < q1.epsilon[0] < 2.8 and q1.delta[0] < 0.2
    assert q1.delta[0] == q0.delta[0] and q1.a[0] != q0.a[0]
    # sun with DHI = 0: clearest category, Delta at its lower bound
    s = ps.perez_parameters(800.0, 0.0, 30.0)
    assert s.category[0] == 8 and s.delta[0] == 0.01


def test_overcast_category_formula():
    # category 1: c = exp((Delta (x1 + x2 Z))^x3) - x4, d = -exp(Delta (x1 + x2 Z)) + x3 + Delta x4
    p = ps.perez_parameters(0.0, 200.0, np.degrees(0.6), clamp=False, brightness_correction=False)
    delta = 200.0 * ps.airmass(np.degrees(0.6)) / 1361.0
    c, d = ps.LUMINANCE_1993[0, 2], ps.LUMINANCE_1993[0, 3]
    assert p.category[0] == 1
    assert p.c[0] == pytest.approx(np.exp((delta * (c[0] + c[1] * 0.6)) ** c[2]) - c[3])
    assert p.d[0] == pytest.approx(-np.exp(delta * (d[0] + d[1] * 0.6)) + d[2] + delta * d[3])


def test_luminous_efficacy_formulas():
    p = ps.perez_parameters([600.0, 50.0], [100.0, 300.0], [40.0, 70.0])
    kg, kd, kb = ps.luminous_efficacy(p, [40.0, 70.0], precipitable_water=1.5)
    for i, z in enumerate([40.0, 70.0]):
        j, zr = p.category[i] - 1, np.radians(z)
        a = ps.EFFICACY_DIFFUSE[j]
        assert kd[i] == pytest.approx(a[0] + a[1] * 1.5 + a[2] * np.cos(zr) + a[3] * np.log(p.delta[i]))
        b = ps.EFFICACY_DIRECT[j]
        assert kb[i] == pytest.approx(max(b[0] + b[1] * 1.5 + b[2] * np.exp(5.73 * zr - 5) + b[3] * p.delta[i], 0))
    assert np.all((kg > 80) & (kg < 140)) and np.all((kd > 90) & (kd < 160))

"""Validation of perezsky; writes results/validation.md.

1. Parameters and luminous efficacy against Radiance's gendaylit (reference implementation),
   for 5 solar altitudes x 9 irradiance conditions x 2 azimuths.
2. Plane irradiance of the discrete sky against rtrace on the exported continuous sky.
3. Annual plane-of-array irradiance against pvlib (Perez 1990 transposition, isotropic model)
   for pvlib's bundled TMY3 file (informational: different models).

Usage:  python validate.py        (needs Radiance: RADIANCE_DIR or Radiance/bin on PATH; pvlib)
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import perezsky as ps

HERE = Path(__file__).parent
WHITE_EFFICACY = 179.0
GENDAYLIT = dict(day_of_year=0, solar_constant=1367.0, airmass_model="kasten1966", limit_gradation=False)
ALTITUDES = [5.0, 15.0, 35.0, 60.0, 80.0]
IRRADIANCE = [(0.0, 100.0), (20.0, 250.0), (150.0, 150.0), (300.0, 150.0), (500.0, 120.0),
              (700.0, 100.0), (850.0, 60.0), (900.0, 30.0), (1.6, 681.0)]


def radiance_env():
    env = os.environ.copy()
    rdir = env.get("RADIANCE_DIR")
    if rdir:
        env["PATH"] = str(Path(rdir) / "bin") + os.pathsep + env["PATH"]
        env["RAYPATH"] = os.pathsep.join([".", str(Path(rdir) / "lib")])
    if shutil.which("gendaylit", path=env["PATH"]) is None:
        sys.exit("Radiance not found: set RADIANCE_DIR")
    return env


def run(env, cmd, stdin=None, cwd=None):
    exe = shutil.which(cmd[0], path=env["PATH"])
    p = subprocess.run([exe] + [str(c) for c in cmd[1:]], input=stdin, cwd=cwd, env=env, capture_output=True)
    return p.stdout


def gendaylit(env, alt, az, dni, dhi, output):
    out = run(env, ["gendaylit", "-ang", alt, az, "-W", dni, dhi, "-O", output]).decode()
    hdr = re.search(r"precipitable water content : ([\d.]+) ([\d.]+)", out)
    if hdr is None:
        return None
    vals = [float(v) for v in out.split("perezlum.cal")[1].split()[2:12]]
    sun = re.search(r"void light solar\s+0\s+0\s+3 (\S+)", out)
    return tuple(map(float, hdr.groups())), vals, (float(sun.group(1)) if sun else 0.0)


def check_gendaylit(env):
    dpar, deff_d, deff_b, refused, n = 0.0, 0.0, 0.0, [], 0
    for alt, az, (dni, dhi) in itertools.product(ALTITUDES, (-40.0, 30.0), IRRADIANCE):
        r1 = gendaylit(env, alt, az, dni, dhi, 1)
        p = ps.perez_parameters(dni, dhi, 90.0 - alt, **GENDAYLIT)
        if r1 is None:
            if (alt, dni, dhi, float(p.b[0])) not in refused:      # same for both azimuths
                refused.append((alt, dni, dhi, float(p.b[0])))
            continue
        r0 = gendaylit(env, alt, az, dni, dhi, 0)
        n += 1
        dpar = max(dpar, float(np.max(np.abs(p.as_array()[:, 0] - r1[1][2:7]))))
        _, kd, kb = ps.luminous_efficacy(p, 90.0 - alt)
        if dhi > 0:
            deff_d = max(deff_d, abs(kd[0] / (r0[1][0] / r1[1][0] * WHITE_EFFICACY) - 1))
        if r1[2] > 0:
            deff_b = max(deff_b, abs(kb[0] - r0[2] / r1[2] * WHITE_EFFICACY))
    return n, dpar, deff_d, deff_b, refused


SKIES = [("clear, sun SE", 850.0, 90.0, 35.0, 140.0), ("hazy, sun W", 400.0, 220.0, 55.0, 250.0),
         ("overcast", 0.0, 180.0, 60.0, 180.0), ("low sun NW", 300.0, 60.0, 84.0, 300.0),
         ("bright overcast", 1.6, 681.0, 48.6, 184.0)]
PLANES = [(0, 0), (30, 180), (60, 180), (90, 180), (90, 90), (90, 0), (90, 270), (45, 135), (120, 180)]


def check_rtrace(env):
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, dni, dhi, zen, azi in SKIES:
            sky = ps.perez_sky(dni, zen, azi, dhi=dhi, grid="medium")
            ps.write_radiance_sky(sky, tmp / "sky.rad", ground_albedo=0.3)
            (tmp / "sky.oct").write_bytes(run(env, ["oconv", "sky.rad"], cwd=tmp))
            normals = np.array([ps.surface_normal(t, a) for t, a in PLANES])
            rays = "\n".join(f"0 0 0 {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}" for n in normals) + "\n"
            out = run(env, ["rtrace", "-h", "-I", "-ab", "1", "-aa", "0", "-ad", "65536", "-as", "0",
                            "-lw", "1e-7", "sky.oct"], stdin=rays.encode(), cwd=tmp).decode()
            rt = np.array(out.split(), dtype=float).reshape(-1, 3) @ np.array([0.265, 0.670, 0.065])
            for grid in ("low", "medium"):
                s = ps.perez_sky(dni, zen, azi, dhi=dhi, grid=grid)
                ref = s.plane_irradiance([t for t, _ in PLANES], [a for _, a in PLANES], 0.3)["total"][0]
                rows.append((name, grid, float(np.max(np.abs(ref / rt - 1)))))
    return rows


def check_pvlib():
    import pvlib
    data, meta = pvlib.iotools.read_tmy3(Path(pvlib.__file__).parent / "data" / "723170TYA.CSV",
                                         map_variables=True)
    loc = pvlib.location.Location(meta["latitude"], meta["longitude"], altitude=meta["altitude"])
    # mid-interval sun position (hour-ending TMY3 data)
    sp = loc.get_solarposition(data.index - np.timedelta64(30, "m"))
    zen, az = sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy()
    dni, dhi, ghi = (data[c].to_numpy(dtype=float) for c in ("dni", "dhi", "ghi"))
    doy = data.index.dayofyear.to_numpy()
    dni_extra = pvlib.irradiance.get_extra_radiation(doy, solar_constant=1361.0)
    am = pvlib.atmosphere.get_relative_airmass(zen)
    planes = [(20, 180), (35, 180), (60, 180), (90, 180), (90, 90), (90, 270), (90, 0)]
    tilt, azim = np.array(planes).T
    ours = ps.poa_irradiance(tilt, azim, dni, zen, az, dhi=dhi, albedo=0.2, day_of_year=doy, grid="low")
    rows = []
    for k, (t, a) in enumerate(planes):
        res = {}
        for model in ("perez", "isotropic"):
            kw = dict(dni_extra=dni_extra, airmass=am) if model == "perez" else {}
            tot = pvlib.irradiance.get_total_irradiance(t, a, zen, az, dni, ghi, dhi, albedo=0.2,
                                                        model=model, **kw)
            res[model] = np.nansum(tot["poa_global"]) / 1000
        mine = ours["total"][:, k].sum() / 1000
        rows.append((t, a, mine, res["perez"], res["isotropic"]))
    return meta, rows


def main():
    env = radiance_env()
    n, dpar, deff_d, deff_b, refused = check_gendaylit(env)
    rt = check_rtrace(env)
    meta, pv = check_pvlib()
    lines = ["# perezsky validation", "",
             "Generated by `validation/validate.py`.", "",
             "## 1. Parameters and luminous efficacy vs. Radiance gendaylit", "",
             f"{n} sky conditions compared (solar altitude 5-80 deg, overcast to clear, two azimuths), "
             "with gendaylit's settings (Kasten 1966 air mass, E0 = 1367 W/m2, day number 0 with `-ang`, "
             "no cap on b).", "",
             f"- Max. absolute difference of the parameters a..e: {dpar:.1e} (gendaylit prints 6 decimals)",
             f"- Max. relative difference of the diffuse luminous efficacy: {deff_d:.1e} "
             "(gendaylit prints the normalisation with 4 significant digits)",
             f"- Max. absolute difference of the direct luminous efficacy: {deff_b:.2f} lm/W", "",
             f"gendaylit refused {len(refused)} conditions (\"positive Perez parameter B, printing error sky\", "
             "i.e. a black sky). perezsky computes b > 0 for exactly these and caps it at 0 by default:", ""]
    lines += [f"- altitude {a:.0f} deg, DNI {d:.0f}, DHI {h:.0f} W/m2: b = {b:+.4f}" for a, d, h, b in refused]
    lines += ["", "## 2. Plane irradiance vs. rtrace on the exported sky", "",
              f"{len(SKIES)} skies x {len(PLANES)} planes (tilt 0-120 deg, all orientations), albedo 0.3. "
              "rtrace: `-ab 1 -aa 0 -ad 65536 -as 0`. Max. relative deviation per sky:", "",
              "| Sky | grid | max. deviation |", "|---|---|---|"]
    lines += [f"| {name} | {grid} | {d:.2%} |" for name, grid, d in rt]
    lines += ["", "## 3. Annual plane-of-array irradiation vs. pvlib (informational)", "",
              f"pvlib's TMY3 file for {meta['Name']} ({meta['latitude']:.2f} N), albedo 0.2, hourly, "
              "sun at mid-interval. pvlib's Perez model is the 1990 *transposition* model, a different "
              "(empirical) model from the 1993 sky distribution used here.", "",
              "| tilt | azimuth | perezsky [kWh/m2] | pvlib Perez 1990 | pvlib isotropic |",
              "|---|---|---|---|---|"]
    lines += [f"| {t} | {a} | {m:.0f} | {p:.0f} ({m / p - 1:+.1%}) | {i:.0f} ({m / i - 1:+.1%}) |"
              for t, a, m, p, i in pv]
    out = HERE / "results" / "validation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

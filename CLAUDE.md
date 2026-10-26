# perezsky — notes for Claude sessions

Python package for the Perez (1993) all-weather sky: radiance/luminance distribution on a
discretised hemisphere, normalised to measured DNI/DHI. Also covers plane irradiance (incl.
long time series), horizon masking with a terrain model, Perez (1990) luminous efficacy and
export to Radiance. Validated against Radiance's gendaylit and rtrace. Extracted from the
slopevf project (`C:\Users\wimm\Documents\GitHub\slopevf`), where the same model was corrected
relative to the user's older MATLAB code. `README.md` is the full user documentation.

Released: PyPI `perezsky` 1.0.0, https://github.com/WildSolar/PerezSky, Zenodo concept DOI
10.5281/zenodo.23666376 (version 1.0.0: 10.5281/zenodo.23666377). Cite the version DOI in
publications.

## Using it from another project

Declare it in that project's `pyproject.toml` (editable, so it follows this repository), then
`uv sync`:

```toml
[project]
dependencies = ["perezsky[plot]"]

[tool.uv.sources]
perezsky = { path = "C:/Users/wimm/Documents/packages/PerezSky", editable = true }
```

To freeze a finished project, or for code that others install, depend on a released version
from PyPI instead (`"perezsky>=1.0"`, without the uv source). Don't copy the code into the
other project.

```python
import perezsky as ps

# angles in degrees; azimuth from NORTH, clockwise (east = 90); DNI/DHI/GHI in W/m2
sky = ps.perez_sky(dni, solar_zenith, solar_azimuth, dhi=dhi,   # or ghi=ghi
                   day_of_year=doy,                  # date-dependent E0 (optional)
                   grid="medium",                    # "low" | "medium" | "high" | "ultra"
                   sun_model="point",                # "point" (beam separate) | "buie" (circumsolar)
                   horizon=None,                     # ps.Horizon(...), see below
                   quantity="radiance")              # or "luminance" (cd/m2, lux)
sky.L, sky.beam, sky.ghi                             # (n_cells, n_t) radiance, (n_t,) point beam, local GHI
sky.plane_irradiance(tilt, azimuth, albedo)          # dict "beam","sky","ground","total": (n_t, n_planes)

out = ps.poa_irradiance(tilt, azimuth, dni, zen, az, dhi=dhi, albedo=0.2, day_of_year=doy)  # chunked series

hor = ps.Horizon.from_file("site.hor")               # .hor files: SOUTH = 0 (default convention there)
hor = ps.Horizon.from_horizontool(ht_horizon, albedo=0.3, terrain="facing")
p = ps.perez_parameters(dni, dhi, zen)               # epsilon, delta, category, a..e
ps.write_radiance_sky(sky, "sky.rad", index=0)       # writes sky.rad + sky.cal (point sun, no horizon)
```

Pitfalls:

- **Azimuth conventions.** The API uses north = 0, clockwise (as pvlib and the horizontool
  API). `.hor` files use south = 0: `Horizon.from_file` assumes that, `Horizon(...)` assumes
  north unless `convention="south"`. slopevf uses south = 0, positive towards west; convert with
  `az_north = az_south + 180`.
- **Night and memory.** `perez_sky` requires the sun above the horizontal and holds
  n_cells × n_t floats. For long series use `poa_irradiance`, which chunks and returns 0 at
  night.
- **Defaults keep the model in its range.** `clamp`, `brightness_correction` and
  `limit_gradation` are on by default.
  - Without `limit_gradation`, very bright overcast skies, and clear skies with a sun below
    about 5°, can give b > 0, and vertical planes then receive many times the GHI.
  - gendaylit-identical parameters: `airmass_model="kasten1966", solar_constant=1367,
    limit_gradation=False`. gendaylit also uses day number 0 with `-ang`.
- **`sun_model`.** `"point"` is the right choice for plane irradiance. `"buie"` spreads the
  beam over a 6° cone; use it only when angular acceptance matters (concentrators,
  circumsolar studies).
- **Horizon.** DNI/DHI are treated as free-horizon values. The terrain model `"facing"`
  (default) is better than `"horizontal"`; it was validated in slopevf.
- **Ground term.** `plane_irradiance` treats the ground as an isotropic reflector of the local
  GHI, the standard approach. Shading by rows or other obstacles is not included; use a
  dedicated model for that, e.g. slopevf for PV arrays.

## Working on this repository

- Releases: bump the version in `pyproject.toml`, `src/perezsky/__init__.py` and
  `CITATION.cff` together; the user publishes a GitHub release `vX.Y.Z`, which Zenodo archives
  and `.github/workflows/publish.yml` publishes to PyPI (trusted publishing). Then add the new
  version DOI to `CITATION.cff` and the README citation.
- Tests: `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider`. The `radiance` tests
  compare with gendaylit/rtrace and need `RADIANCE_DIR` (here `C:\Users\wimm\tools\Radiance`);
  without it they are skipped. Run the full suite with Radiance before committing.
- Model or numerical changes: rerun `validation/validate.py` (needs Radiance and pvlib) and
  commit the updated `validation/results/validation.md`. The gendaylit comparison must stay
  within print precision.
- Coefficients are transcribed from gendaylit.c, the reference implementation;
  `tests/test_model.py` holds an independent transcription.
- git is not on PATH: use GitHub Desktop's bundled
  `%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe`. Commit on `main` with a
  message file (`commit -F`, UTF-8 without BOM); never push — the user pushes from GitHub Desktop.

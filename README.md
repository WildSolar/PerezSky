# perezsky

The Perez (1993) all-weather sky as a radiance (or luminance) distribution on a discretised
hemisphere, normalised to measured irradiance. On top of it:

- irradiance on arbitrarily oriented planes, also for long time series;
- masking by a local horizon, with a model of the terrain radiance;
- luminance and illuminance via the Perez (1990) luminous efficacy models;
- export of the same sky to Radiance;
- a polar plot of the sky.

The implementation is validated against Radiance's `gendaylit` and `rtrace`
([validation/results/validation.md](https://github.com/WildSolar/PerezSky/blob/main/validation/results/validation.md)).

```bash
pip install perezsky              # with plotting: pip install "perezsky[plot]"
```

## Quick start

```python
import perezsky as ps

# one or many time steps; angles in degrees, azimuth from north, clockwise
sky = ps.perez_sky(dni=[850, 120], solar_zenith=[35, 70], solar_azimuth=[140, 250],
                   dhi=[90, 200], day_of_year=[172, 172])
sky.L                      # (n_cells, n_t) sky radiance [W/m2/sr]
sky.beam                   # (n_t,) DNI of the point sun [W/m2] (0 while hidden by a horizon)
res = sky.plane_irradiance(tilt=[30, 90], azimuth=[180, 90], albedo=0.2)
res["total"]               # (n_t, n_planes); also "beam", "sky", "ground"

# a year of hourly data in chunks (sun below the horizontal -> 0)
out = ps.poa_irradiance(35, 180, dni, solar_zenith, solar_azimuth, dhi=dhi,
                        albedo=0.2, day_of_year=doy)
```

## Model

**Sky condition.** For each time step:

- the sky clearness ε = ((DHI + DNI)/DHI + 1.041 Z³) / (1 + 1.041 Z³);
- the sky brightness Δ = DHI · m / E₀, with the relative air mass m and the extraterrestrial
  irradiance E₀;
- one of eight clearness categories, determined by ε.

**Distribution.** The relative distribution is

L ∝ (1 + a exp(b / cos ζ)) · (1 + c exp(d γ) + e cos² γ),

with ζ the zenith angle of the sky element and γ its angle to the sun. The parameters a..e
come from the coefficients of Perez et al. (1993); for the overcast category, c and d use
their own formulas.

**Normalisation.**

- The sky cells above the horizontal integrate exactly to DHI, for any grid resolution.
- With `sun_model="point"` (default), the beam DNI is kept as a separate point source in the
  exact sun direction. For irradiance on planes, this is more accurate than spreading it over
  sky cells.
- With `sun_model="buie"`, the beam is distributed over a circumsolar cone (default 6°)
  following the Buie et al. (2003) sunshape. Inside the cone it replaces the Perez radiance and
  integrates to DNI cos Z. This matters when angular acceptance is relevant, e.g. for
  concentrators.

**Range of validity.** By default the inputs are kept within the range of the model:

- `clamp=True`: ε ∈ [1, 12.01) and Δ ∈ [0.01, 0.6], as gendaylit does.
- `brightness_correction=True`: Δ ≥ 0.2 for 1.065 < ε < 2.8 when computing a..e, as
  gendaylit does. gendaylit marks it as a correction of the Perez model, without a
  reference; see the section below.
- `limit_gradation=True`: b ≤ 0.
  - For very bright overcast skies (e.g. cloud enhancement over snow, GHI ≈ 700 W/m² with
    DNI ≈ 0) and for clear skies with a very low sun, the model gives b > 0.
  - The factor exp(b / cos ζ) then grows without bound towards the horizon. Vertical planes
    received up to 30 times the GHI in a real data set.
  - gendaylit refuses such skies and outputs a black sky ("positive Perez parameter B");
    perezsky caps b at 0 instead.

**The brightness correction.** It changes only the *shape* of the distribution; the
normalisation to the actual DHI stays the same.

- **Where it acts:** on dark skies of intermediate clearness (thin or broken cloud with little
  diffuse light, Δ < 0.2 for 1.065 < ε < 2.8).
- **Why it is needed:** the parameters are linear in Δ and were fitted to measured skies. For
  low Δ in categories 2–5 they extrapolate to a gradation 1 + a·exp(b/cos ζ) with
  a·exp(b) ≈ −1. The zenith then becomes almost black, and the horizon up to 3.5 times
  brighter than the average sky radiance DHI/π. Real skies of this type do not look like that.
- **What it does:** raising Δ to 0.2 makes b more negative. The gradation then approaches 1
  only close to the horizon, which gives a moderately brighter horizon and a zenith of 0.4–1.0
  times the average radiance.
- **Size of the effect:** in such hours the sky-diffuse irradiance on vertical planes drops by
  a median of 4–15% (up to about 45%). In the opposite direction, a few very dark skies
  (Δ ≈ 0.05) gain up to 25%.
- **How often:** such hours make up 1–2% of the daylight hours (Davos PVGIS TMY, Greensboro
  TMY3). Annual sums change by at most 0.2% of the sky-diffuse and 0.1% of the total
  irradiation, on any plane.

See [validation/results/brightness_correction.md](https://github.com/WildSolar/PerezSky/blob/main/validation/results/brightness_correction.md).

**Other settings.**

- **Air mass:** Kasten & Young (1989) by default; `airmass_model="kasten1966"` selects
  gendaylit's formula.
- **E₀:** the solar constant of 1361 W/m², corrected for the Earth–Sun distance (Spencer 1971)
  if `day_of_year` is given; gendaylit uses 1367 W/m².

**Grids.** Elevation bands of 0.5°, each divided into azimuthal cells of near-constant size,
with exact solid angles:

| Resolution | Cells |
|---|---|
| `low` | 10'401 |
| `medium` | 20'716 |
| `high` | 41'335 |
| `ultra` | 82'594 |

Plane irradiance on `low` is within 0.3% of `ultra`. `poa_irradiance` uses `low` by default.

## Planes and ground

`Sky.plane_irradiance(tilt, azimuth, albedo)` sums three parts:

- **beam:** DNI · cos(incidence), zero while the sun is behind the plane or the horizon;
- **sky:** the sky cells in front of the plane, weighted with the cosine;
- **ground:** isotropic reflection of the local global horizontal irradiance below the
  horizontal, albedo · GHI · (1 − cos tilt)/2.

## Horizon

```python
hor = ps.Horizon.from_file("site.hor")                     # PVsyst/horizontool .hor: south = 0
hor = ps.Horizon(azimuth, elevation, convention="north")   # arrays, north = 0
hor = ps.Horizon.from_horizontool(ht.compute_horizon(lat, lon), albedo=0.3)
sky = ps.perez_sky(..., horizon=hor)
```

**Masking.** Sky cells below the horizon get the radiance of the terrain, and the beam is
blocked while the sun is behind the horizon. DNI and DHI are treated as free-horizon values.
If they were measured on site behind the horizon, this is still a good approximation for the
sky: the masked part is replaced anyway.

**Terrain models.**

| `terrain=` | Model | Deviation from ray-traced terrain |
|---|---|---|
| `"facing"` (default) | Slopes of 30° facing the observer: beam (if the sun is visible from the observer), isotropic sky, own surroundings | ≤ 2% on tilted planes |
| `"horizontal"` | albedo · GHI / π | up to 14% |

The deviations were determined with Radiance in the slopevf project. Negative horizon
elevations are treated as 0.

## Luminance

`perez_sky(..., quantity="luminance")` gives luminance [cd/m²] and illuminance [lux].

- **Conversion:** DNI and DHI are converted with the Perez et al. (1990) luminous efficacies
  of the direct and diffuse components (`precipitable_water` in cm, default 2 as in
  gendaylit), and the same distribution is normalised to them.
- **Efficacies:** `luminous_efficacy(params, zenith)` returns the global, diffuse and direct
  efficacies [lm/W] directly.

## Radiance export

```python
ps.write_radiance_sky(sky, "sky.rad", index=0, ground_albedo=0.2)   # writes sky.rad and sky.cal
```

The exported sky differs from gendaylit's in two ways:

- **No horizon blending.** gendaylit's `perezlum.cal` mixes about 45% ground brightness into
  the sky at the horizon, which changes vertical-plane irradiance by up to 5%; the exported sky
  has no such blending.
- **Same normalisation.** It is normalised to the same DHI as the discrete sky.

`rtrace` on the exported sky reproduces `Sky.plane_irradiance` within 0.1%. Only skies with
`sun_model="point"` and without horizon can be exported.

## Corrections relative to earlier MATLAB implementations

The MATLAB code this package replaces (`skyLuminanceBatch.m`, `totalSkyLuminance.m`) had
several deviations from the published model. All of them are corrected here:

- Three coefficients were mistyped:
  - `x4` of b in category 1: −1.233 instead of −0.1233;
  - `x3` of c in category 6: −7.1407 instead of −7.1406;
  - `x4` of d in category 7: 0.2652 instead of 0.2656.
- The category-1 formula for c raised only (x1 + x2 Z) to the power x3 instead of
  Δ (x1 + x2 Z): exp(Δ (x1 + x2 Z)^x3) instead of exp((Δ (x1 + x2 Z))^x3).
- The air mass function was called with the elevation in radians instead of sin(elevation),
  which gives complex values above 1 rad.
- The model ran without range limits.

## Validation

`validation/validate.py` (needs Radiance and pvlib) writes
[validation/results/validation.md](https://github.com/WildSolar/PerezSky/blob/main/validation/results/validation.md):

- **Parameters and luminous efficacies vs. gendaylit:** 86 sky conditions, solar altitudes
  5–80°, overcast to clear. Agreement within gendaylit's print precision:
  - parameters a..e to 2.5·10⁻⁶;
  - diffuse efficacy to 0.06%;
  - direct efficacy to 0.04 lm/W.

  gendaylit refuses 2 of the conditions as error skies, exactly those with b > 0.
- **Plane irradiance vs. rtrace:** 5 skies × 9 planes on the exported sky, within 0.1%.
- **Annual POA vs. pvlib (TMY3, Greensboro):** within about 1% of pvlib's Perez (1990)
  transposition model for south-facing and vertical planes. The isotropic model deviates by up
  to 15% (vertical north).

The tests (`pytest`) include the gendaylit and rtrace comparisons, marked `radiance` and skipped
without Radiance.

## Citation

If you use perezsky in published work, please cite the archived version on Zenodo (see
[CITATION.cff](https://github.com/WildSolar/PerezSky/blob/main/CITATION.cff); the DOI badge is added with the first release).

## License

MIT, see [LICENSE](https://github.com/WildSolar/PerezSky/blob/main/LICENSE).

## References

- Perez, R., Seals, R., Michalsky, J. (1993). All-weather model for sky luminance
  distribution — preliminary configuration and validation. Solar Energy 50(3), 235–245.
- Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R. (1990). Modeling daylight
  availability and irradiance components from direct and global irradiance. Solar Energy 44(5),
  271–289.
- Buie, D., Monger, A. G., Dey, C. J. (2003). Sunshape distributions for terrestrial solar
  simulations. Solar Energy 74(2), 113–122.
- Kasten, F., Young, A. T. (1989). Revised optical air mass tables and approximation formula.
  Applied Optics 28(22), 4735–4738.
- Spencer, J. W. (1971). Fourier series representation of the position of the sun. Search 2(5).

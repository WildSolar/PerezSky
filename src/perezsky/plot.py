"""Plot of a sky distribution (requires the ``plot`` extra, i.e. matplotlib)."""

from __future__ import annotations

import numpy as np


def plot_sky(sky, index: int = 0, ax=None, log: bool = True, cmap: str = "viridis"):
    """Polar plot (zenith in the centre, north up, east right) of the sky cells of time step
    ``index``. Returns the matplotlib axes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    g = sky.grid
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    values = sky.L[:, index]
    norm = LogNorm(vmin=max(values[values > 0].min(), values.max() * 1e-3), vmax=values.max()) if log else None
    sc = ax.scatter(np.radians(g.azimuth), 90.0 - g.elevation, c=values, s=2, cmap=cmap, norm=norm,
                    linewidths=0)
    if sky.beam[index] > 0:
        ax.plot(np.radians(sky.sun_azimuth[index]), 90.0 - sky.sun_elevation[index], "o",
                mfc="none", mec="red", ms=10)
    ax.set_ylim(0, 90)
    ax.set_yticks([30, 60, 90])
    ax.set_yticklabels(["60°", "30°", "0°"])
    unit = "cd/m²" if sky.quantity == "luminance" else "W/m²/sr"
    plt.colorbar(sc, ax=ax, label=unit, shrink=0.8)
    return ax

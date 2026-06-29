"""
Interactive maps with Plotly: gridded climate fields as heatmaps with India
state boundaries overlaid. Renders client-side (fast, smooth on slider changes),
no map tiles / tokens / websocket round-trips - reliable inside Streamlit.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
from viz import theme

_GEOJSON = os.path.join(C.DATA_DIR, "geojson", "india_states.geojson")
_BOUNDARY = None  # cached (lon, lat) polyline with None separators


def _to_plotly_colorscale(cmap, n=32):
    return [[i / (n - 1), f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"]
            for i, (r, g, b, _) in ((i, cmap(i / (n - 1))) for i in range(n))]


def india_boundary(step=70):
    """Return (lon, lat) arrays tracing all state borders, None-separated.

    Polygon rings are decimated (keep every `step`-th vertex) so the overlay
    stays light enough for smooth interactive rendering.
    """
    global _BOUNDARY
    if _BOUNDARY is not None:
        return _BOUNDARY
    lon, lat = [], []
    try:
        gj = json.load(open(_GEOJSON))
        for feat in gj["features"]:
            geom = feat["geometry"]
            polys = geom["coordinates"]
            if geom["type"] == "Polygon":
                polys = [polys]
            for multi in polys:
                for ring in multi:
                    if len(ring) < 4:
                        continue
                    pts = ring[::step]
                    if pts[-1] != ring[-1]:
                        pts = pts + [ring[-1]]      # close the ring
                    for x, y in pts:
                        lon.append(x); lat.append(y)
                    lon.append(None); lat.append(None)
    except Exception:
        pass
    _BOUNDARY = (lon, lat)
    return _BOUNDARY


def crop_to_bounds(field, lats, lons, bounds):
    la = (lats >= bounds["lat_min"]) & (lats <= bounds["lat_max"])
    lo = (lons >= bounds["lon_min"]) & (lons <= bounds["lon_max"])
    return field[np.ix_(la, lo)], lats[la], lons[lo]


def field_figure(field, lats, lons, key, title, landmask=None,
                 bounds=None, height=560, vmin=None, vmax=None, unit=None):
    """Build a Plotly heatmap figure for a 2-D field with state boundaries."""
    import plotly.graph_objects as go

    z = np.array(field, dtype="float64")
    if landmask is not None:
        z = np.where(landmask, z, np.nan)          # ocean transparent
    dvmin, dvmax = theme.VRANGE.get(key, (np.nanmin(z), np.nanmax(z)))
    vmin = dvmin if vmin is None else vmin
    vmax = dvmax if vmax is None else vmax
    cmap = theme.CMAPS.get(key, theme.TEMP)
    unit = unit if unit is not None else theme.LABELS.get(key, key)

    fig = go.Figure(go.Heatmap(
        z=z, x=lons, y=lats,
        colorscale=_to_plotly_colorscale(cmap),
        zmin=vmin, zmax=vmax,
        colorbar=dict(title=dict(text=unit, side="right"), thickness=14, len=0.9,
                      tickcolor=theme_text(), tickfont=dict(color=theme_text())),
        hovertemplate="lat %{y:.2f}°N<br>lon %{x:.2f}°E<br>%{z:.1f}<extra></extra>",
    ))
    blon, blat = india_boundary()
    if blon:
        fig.add_trace(go.Scattergl(x=blon, y=blat, mode="lines",
                                   line=dict(color="rgba(180,200,255,0.5)", width=0.8),
                                   hoverinfo="skip", showlegend=False))
    xr = [bounds["lon_min"], bounds["lon_max"]] if bounds else [float(lons.min()), float(lons.max())]
    yr = [bounds["lat_min"], bounds["lat_max"]] if bounds else [float(lats.min()), float(lats.max())]
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme_text(), size=15)),
        height=height, margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor=C.PALETTE["bg"], plot_bgcolor=C.PALETTE["bg"],
        xaxis=dict(range=xr, color=theme_text(), gridcolor="rgba(255,255,255,0.05)",
                   constrain="domain"),
        yaxis=dict(range=yr, color=theme_text(), gridcolor="rgba(255,255,255,0.05)",
                   scaleanchor="x", scaleratio=1.0),
        dragmode="pan",
    )
    return fig


def theme_text():
    return C.PALETTE["text"]


def line_figure(x, series, title, ytitle, height=300, colors=None):
    """Dark-themed multi-series line chart. `series` = {name: y-values}."""
    import plotly.graph_objects as go
    palette = colors or ["#22D3EE", "#FF7B00", "#A7F3D0", "#FDE047", "#FF2D55"]
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=list(x), y=list(y), mode="lines+markers", name=name,
                                 line=dict(color=palette[i % len(palette)], width=2.5),
                                 marker=dict(size=6)))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme_text(), size=14)),
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor=C.PALETTE["bg"], plot_bgcolor=C.PALETTE["panel"],
        font=dict(color=theme_text()),
        xaxis=dict(title=dict(text="forecast lead day"), gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=dict(text=ytitle), gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    )
    return fig


def bar_figure(categories, series, title, ytitle, height=320, colors=None):
    """Dark-themed grouped bar chart. `series` = {name: values aligned to categories}."""
    import plotly.graph_objects as go
    palette = colors or ["#22D3EE", "#FF7B00", "#A7F3D0"]
    fig = go.Figure()
    for i, (name, y) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=list(categories), y=list(y), name=name,
                             marker_color=palette[i % len(palette)]))
    fig.update_layout(
        title=dict(text=title, font=dict(color=theme_text(), size=14)),
        height=height, margin=dict(l=10, r=10, t=40, b=10), barmode="group",
        paper_bgcolor=C.PALETTE["bg"], plot_bgcolor=C.PALETTE["panel"],
        font=dict(color=theme_text()),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(title=dict(text=ytitle), gridcolor="rgba(255,255,255,0.07)"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    )
    return fig

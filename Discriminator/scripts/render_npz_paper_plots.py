"""Render schema-v2 plot NPZ bundles without the original experiment run."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

try:
    from .plot_bundles import configure_plot_bundle_saving, save_figure_bundle
except ImportError:
    from plot_bundles import configure_plot_bundle_saving, save_figure_bundle


def load_bundle(path):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    if int(metadata.get("schema_version", 1)) < 2:
        archive.close()
        raise ValueError(
            f"{path} uses legacy schema {metadata.get('schema_version', 1)}. "
            "Render its complete run with render_paper_plots.py instead; legacy "
            "bundles lack reliable labels, scales, and map normalization."
        )
    return archive, metadata


def make_axis(figure, metadata):
    projection = None
    if "GeoAxes" in str(metadata.get("projection", "")):
        try:
            import cartopy.crs as ccrs
            projection = ccrs.PlateCarree()
        except ImportError:
            pass
    kwargs = {"projection": projection} if projection is not None else {}
    return figure.add_axes(metadata.get("position", [0.12, 0.12, 0.8, 0.8]), **kwargs)


def reconstruct(archive, metadata, width):
    original = metadata.get("figure_size_inches", [width, width * 0.65])
    height = width * float(original[1]) / max(float(original[0]), 1e-12)
    figure = plt.figure(figsize=(width, height))
    for axis_metadata in metadata.get("axes", []):
        axis = make_axis(figure, axis_metadata)
        for line in axis_metadata.get("lines", []):
            prefix = line["prefix"]
            marker = line.get("marker")
            axis.plot(
                archive[f"{prefix}_x"], archive[f"{prefix}_y"],
                label=line.get("label"), color=line.get("color"),
                linestyle=line.get("linestyle", "-"),
                marker=None if marker in {"None", "none", ""} else marker,
            )
        for collection in axis_metadata.get("collections", []):
            prefix = collection["prefix"]
            values_key, coordinates_key = f"{prefix}_values", f"{prefix}_coordinates"
            offsets_key = f"{prefix}_offsets"
            style = {
                "cmap": collection.get("cmap", "viridis"),
                "vmin": collection.get("vmin"), "vmax": collection.get("vmax"),
            }
            if coordinates_key in archive and values_key in archive:
                coordinates, values = archive[coordinates_key], archive[values_key]
                expected = (coordinates.shape[0] - 1, coordinates.shape[1] - 1)
                if values.size == int(np.prod(expected)):
                    values = values.reshape(expected)
                axis.pcolormesh(
                    coordinates[..., 0], coordinates[..., 1], values,
                    shading="auto", rasterized=True, **style,
                )
            elif offsets_key in archive:
                offsets = archive[offsets_key]
                values = archive[values_key] if values_key in archive else None
                colors = values if values is not None and values.size == len(offsets) else None
                axis.scatter(offsets[:, 0], offsets[:, 1], c=colors, s=8, rasterized=True, **style)
        for image in axis_metadata.get("images", []):
            prefix = image["prefix"]
            axis.imshow(
                archive[prefix], extent=archive[f"{prefix}_extent"],
                cmap=image.get("cmap", "viridis"), vmin=image.get("vmin"),
                vmax=image.get("vmax"), origin=image.get("origin", "upper"),
                rasterized=True,
            )
        for patch in axis_metadata.get("patches", []):
            style = {
                "label": patch.get("label"),
                "facecolor": patch.get("facecolor", "none"),
                "edgecolor": patch.get("edgecolor", "none"),
                "linewidth": patch.get("linewidth", 1.0),
                "alpha": patch.get("alpha"),
            }
            if patch.get("geometry", "rectangle") == "path":
                artist = Polygon(archive[f"{patch['prefix']}_vertices"], closed=True, **style)
            else:
                x, y, patch_width, patch_height = map(float, archive[patch["prefix"]])
                artist = Rectangle((x, y), patch_width, patch_height, **style)
            axis.add_patch(artist)
        axis.set_xscale(axis_metadata.get("xscale", "linear"))
        axis.set_yscale(axis_metadata.get("yscale", "linear"))
        axis.set_xlim(axis_metadata.get("xlim", axis.get_xlim()))
        axis.set_ylim(axis_metadata.get("ylim", axis.get_ylim()))
        axis.set(
            title=axis_metadata.get("title", ""),
            xlabel=axis_metadata.get("xlabel", ""),
            ylabel=axis_metadata.get("ylabel", ""),
        )
        if hasattr(axis, "coastlines") and "GeoAxes" in str(axis_metadata.get("projection", "")):
            axis.coastlines(linewidth=0.45)
        handles, labels = axis.get_legend_handles_labels()
        if any(label and not label.startswith("_") for label in labels):
            axis.legend()
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A schema-v2 NPZ or directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--text-width-inches", type=float, default=5.5)
    parser.add_argument("--width", choices=("full", "half"), default="full")
    args = parser.parse_args()
    source = args.source.resolve()
    paths = [source] if source.is_file() else sorted(
        path for path in source.rglob("*.npz") if not path.name.endswith("_notitle.npz")
    )
    if not paths:
        raise FileNotFoundError(f"No plot NPZ bundles found beneath {source}.")
    common = source.parent if source.is_file() else source
    output = (args.output_dir or (common / "paper_from_npz")).resolve()
    configure_plot_bundle_saving(
        profile="paper", paper_width_inches=args.text_width_inches,
        paper_width_kind=args.width,
    )
    for path in paths:
        archive, metadata = load_bundle(path)
        try:
            figure = reconstruct(archive, metadata, args.text_width_inches)
            relative = Path(path.name) if source.is_file() else path.relative_to(common)
            save_figure_bundle(
                figure, output / "plots" / relative.with_suffix(".png"),
                plot_type=f"paper_replot:{metadata.get('plot_type', 'unknown')}",
                metadata={"source_npz": str(path)},
            )
            plt.close(figure)
        finally:
            archive.close()
    print(f"Rendered {len(paths)} semantic NPZ bundle(s) beneath: {output / 'plots' / 'paper'}")


if __name__ == "__main__":
    main()

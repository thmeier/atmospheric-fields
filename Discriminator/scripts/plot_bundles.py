"""Portable PNG/PDF/NPZ bundles for baseline-pipeline figures."""

import json
from pathlib import Path

import numpy as np


def plot_bundle_paths(png_path):
    """Return the three side-by-side files belonging to a rendered figure."""
    png_path = Path(png_path)
    if png_path.suffix.lower() != ".png":
        raise ValueError(f"Plot bundle path must end in .png, got {png_path}")
    return png_path, png_path.with_suffix(".pdf"), png_path.with_suffix(".npz")


def _safe_array(value):
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError("Plot-bundle arrays must not use object dtype.")
    return array


def _artist_arrays(figure):
    """Capture the numerical artists actually rendered by Matplotlib."""
    arrays = {}
    metadata_axes = []
    for axis_index, axis in enumerate(figure.axes):
        metadata_axes.append({
            "title": axis.get_title(), "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(),
        })
        for line_index, line in enumerate(axis.lines):
            prefix = f"axes_{axis_index}_line_{line_index}"
            arrays[f"{prefix}_x"] = _safe_array(line.get_xdata(orig=False))
            arrays[f"{prefix}_y"] = _safe_array(line.get_ydata(orig=False))
        for collection_index, collection in enumerate(axis.collections):
            prefix = f"axes_{axis_index}_collection_{collection_index}"
            if hasattr(collection, "get_offsets"):
                offsets = collection.get_offsets()
                if offsets is not None and np.asarray(offsets).size:
                    arrays[f"{prefix}_offsets"] = _safe_array(offsets)
            if hasattr(collection, "get_array"):
                values = collection.get_array()
                if values is not None:
                    arrays[f"{prefix}_values"] = _safe_array(values)
            if hasattr(collection, "get_coordinates"):
                coordinates = collection.get_coordinates()
                if coordinates is not None:
                    arrays[f"{prefix}_coordinates"] = _safe_array(coordinates)
        for image_index, image in enumerate(axis.images):
            arrays[f"axes_{axis_index}_image_{image_index}"] = _safe_array(image.get_array())
            arrays[f"axes_{axis_index}_image_{image_index}_extent"] = _safe_array(image.get_extent())
        for patch_index, patch in enumerate(axis.patches):
            # Histogram bars and other rendered rectangles.
            if all(hasattr(patch, method) for method in ("get_x", "get_y", "get_width", "get_height")):
                arrays[f"axes_{axis_index}_patch_{patch_index}"] = _safe_array([
                    patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height(),
                ])
    return arrays, metadata_axes


def rasterize_field_artists(figure):
    """Rasterize the artists that make field plots enormous as vectors.

    A 121x240 pcolormesh is ~29k individually drawn patches, and a gallery stacks
    several per figure. As vectors those reach 50-70 MB per PDF and dominate the
    plot stage; embedding them as a bitmap instead is visually identical at print
    resolution. Only meshes and images are rasterized -- axes, text, lines,
    legends and colorbars stay vector, so the PDF remains crisp and searchable.

    Returns the number of artists rasterized.
    """
    from matplotlib.collections import Collection
    from matplotlib.image import AxesImage

    count = 0
    for axis in figure.get_axes():
        for artist in list(axis.collections) + list(axis.images):
            if isinstance(artist, (Collection, AxesImage)) and not artist.get_rasterized():
                artist.set_rasterized(True)
                count += 1
    return count


def save_figure_bundle(figure, png_path, *, plot_type, payload=None, metadata=None,
                       dpi=220, bbox_inches=None):
    """Save one figure as PNG, PDF, and a no-pickle NPZ replot sidecar."""
    png_path, pdf_path, npz_path = plot_bundle_paths(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": int(dpi)}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    figure.savefig(png_path, **save_kwargs)
    # Rasterize after the PNG (which is a bitmap regardless) and before the PDF, so
    # only the PDF path pays for it. dpi is passed explicitly here: it sets the
    # resolution of the embedded raster, and without it savefig would fall back to
    # the figure default and produce a soft image.
    rasterize_field_artists(figure)
    figure.savefig(pdf_path, dpi=int(dpi), bbox_inches=bbox_inches)

    arrays, axes = _artist_arrays(figure)
    for key, value in (payload or {}).items():
        arrays[f"input_{key}"] = _safe_array(value)
    bundle_metadata = {
        "schema_version": 1,
        "plot_type": str(plot_type),
        "png": png_path.name,
        "pdf": pdf_path.name,
        "npz": npz_path.name,
        "axes": axes,
        **(metadata or {}),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(bundle_metadata, sort_keys=True))
    np.savez_compressed(npz_path, **arrays)

    manifest_path = png_path.parent / "plot_data_manifest.json"
    manifest = {"schema_version": 1, "plots": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            pass
    manifest.setdefault("plots", {})[png_path.stem] = {
        "plot_type": str(plot_type), "png": png_path.name,
        "pdf": pdf_path.name, "npz": npz_path.name,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return [png_path, pdf_path, npz_path]

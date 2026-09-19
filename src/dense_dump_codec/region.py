"""Exact radial prefixes for unrefined native logarithmic KHARMA grids."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np


def parameter_sections(text):
    sections, current = {}, None
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        if body.startswith("<") and body.endswith(">"):
            current = body[1:-1]
            if current in sections:
                raise ValueError("Duplicate native parameter section")
            sections[current] = {}
        elif "=" in body and current is not None:
            key, value = map(str.strip, body.split("=", 1))
            if key in sections[current]:
                raise ValueError("Duplicate native parameter")
            sections[current][key] = value
    return sections


def replace_parameters(text, replacements):
    current, changed, output = None, set(), []
    for line in text.splitlines(keepends=True):
        body = line.split("#", 1)[0].strip()
        if body.startswith("<") and body.endswith(">"):
            current = body[1:-1]
        match = re.match(r"(\s*)(\w+)\s*=", line)
        key = (current, match.group(2)) if match else None
        if key in replacements:
            line = f"{match.group(1)}{key[1]} = {replacements[key]}\n"
            changed.add(key)
        output.append(line)
    if changed != set(replacements):
        raise ValueError("Required native grid parameter is missing")
    return "".join(output)


@dataclass(frozen=True)
class RadialRegion:
    radius_max: float
    source_cells: int
    retained_cells: int
    startx1: float
    dx1: float
    stopx1: float
    halo_cells: int
    alignment: int

    @classmethod
    def from_metadata(cls, native, radius_max, *, alignment=32, halo_cells=1):
        if not math.isfinite(radius_max) or radius_max <= 0:
            raise ValueError("Radial maximum must be finite and positive")
        if alignment < 1 or halo_cells < 1:
            raise ValueError("Radial ROI requires alignment and interpolation halo")
        sections = parameter_sections(native["par_text"])
        mesh, coordinates = sections["parthenon/mesh"], sections["coordinates"]
        if (coordinates.get("transform") not in {"fmks", "mks"}
                or coordinates.get("base", "spherical_ks") != "spherical_ks"
                or mesh.get("refinement", "none") != "none"
                or float(mesh.get("x1rat", "1")) != 1):
            raise ValueError("Radial ROI requires an unrefined logarithmic spherical grid")
        dimensions = tuple(int(mesh[f"nx{axis}"]) for axis in (1, 2, 3))
        blocks = native["meshblock_size"]
        order = np.asarray(native["block_order"]).reshape(-1, 3)
        if dimensions[0] != blocks[0] or np.any(order[:, 0] != 0):
            raise ValueError("Radial ROI currently requires one radial MeshBlock")
        if any(size % block for size, block in zip(dimensions, blocks)):
            raise ValueError("Nonuniform native MeshBlock layout")
        expected = {(0, theta, phi)
                    for theta in range(dimensions[1] // blocks[1])
                    for phi in range(dimensions[2] // blocks[2])}
        if set(map(tuple, order)) != expected or len(order) != len(expected):
            raise ValueError("Incomplete or refined native MeshBlock layout")
        start, stop = float(mesh["x1min"]), float(mesh["x1max"])
        spacing = (stop - start) / dimensions[0]
        if not math.isfinite(spacing) or spacing <= 0:
            raise ValueError("Invalid logarithmic radial spacing")
        if math.log(radius_max) < start:
            raise ValueError("Radial ROI does not intersect the source grid")
        right = math.floor((math.log(radius_max) - start) / spacing - 0.5) + 1
        required = max(1, right + 1 + halo_cells)
        retained = min(dimensions[0], math.ceil(required / alignment) * alignment)
        new_stop = stop if retained == dimensions[0] else start + retained * spacing
        candidates = [new_stop]
        for _ in range(4):
            candidates.extend((math.nextafter(min(candidates), -math.inf),
                               math.nextafter(max(candidates), math.inf)))
        exact = [value for value in candidates if (value - start) / retained == spacing]
        if not exact:
            raise ValueError("Cannot preserve exact native coordinate spacing for this ROI")
        return cls(float(radius_max), dimensions[0], retained, start, spacing,
                   exact[0], halo_cells, alignment)

    def metadata(self, native):
        replacements = {
            ("parthenon/mesh", "nx1"): str(self.retained_cells),
            ("parthenon/mesh", "x1max"): repr(self.stopx1),
        }
        sections = parameter_sections(native["par_text"])
        if "nx1" in sections.get("parthenon/meshblock", {}):
            replacements[("parthenon/meshblock", "nx1")] = str(self.retained_cells)
        return {**native, "par_text": replace_parameters(native["par_text"], replacements),
                "meshblock_size": (self.retained_cells, *native["meshblock_size"][1:]),
                "radial_region": self.describe()}

    def describe(self):
        return {**asdict(self), "data_radius_edge": math.exp(self.stopx1),
                "array_fraction": self.retained_cells / self.source_cells,
                "compressed_chunks_are_spatially_indexed": False}


def radial_exceptions(indices, values, source_cells, retained_cells):
    """Remap full-array C-order exception indices into a radial prefix."""
    indices = indices.astype(np.int64)
    radial = indices % source_cells
    selected = radial < retained_cells
    return ((indices[selected] // source_cells) * retained_cells + radial[selected],
            values[selected])

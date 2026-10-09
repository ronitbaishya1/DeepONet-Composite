
# Mac, ordinary Python.
# Reuses the full validated 64V algorithm unmodified.

import json
import os
from pathlib import Path


root = Path(__file__).resolve().parent

original = (
    root
    / "64V_reconstruct_validate_7region_direct.py"
)

if not original.is_file():
    raise FileNotFoundError(
        "Original validated 64V script missing: {}"
        .format(original)
    )


online = (
    root
    / "data"
    / "hybrid_online_7region_rom"
)


required = [
    online / "rom_solution_7region.npz",
]

required += [
    online / "validation_fields" / filename
    for filename in (
        "left_nodes.csv",
        "center_nodes.csv",
        "right_nodes.csv",
        "left_ip.csv",
        "center_ip.csv",
        "right_ip.csv",
        "full_reference_nodes.csv",
        "full_reference_ip.csv",
        "reaction_summary.json",
    )
]


for path in required:
    if not path.is_file():
        raise FileNotFoundError(
            "65I prerequisite missing: {}"
            .format(path)
        )


# Read, but do NOT modify, original 64V.
source = original.read_text(
    encoding="utf-8"
)


if "hybrid_online_7region_direct" not in source:
    raise RuntimeError(
        "Cannot identify original 64V data locations."
    )


if '"online_solution_7region.npz"' not in source:
    raise RuntimeError(
        "Cannot identify original solution filename."
    )


# Only change the paths.
adapted = source.replace(
    "hybrid_online_7region_direct",
    "hybrid_online_7region_rom",
)

adapted = adapted.replace(
    '"online_solution_7region.npz"',
    '"rom_solution_7region.npz"',
)


if "hybrid_online_7region_direct" in adapted:
    raise RuntimeError(
        "Incomplete directory adaptation."
    )


# Run the same reconstruction and evaluation logic
# used for our validated direct hybrid.
os.chdir(root)

exec(
    compile(
        adapted,
        str(original) + " [65I ROM paths]",
        "exec",
    ),
    {
        "__name__": "__main__",
        "__file__": str(original),
    },
)


out = (
    root
    / "results"
    / "hybrid_online_7region_rom_validation"
)


with (
    out / "ip_matching_audit.json"
).open() as f:
    audit = json.load(f)


if (
    not audit["one_to_one"]
    or audit["maximum_coordinate_distance_mm"] > 1e-4
):
    raise RuntimeError(
        "65I integration-point matching failed."
    )


print("STEP 65I MAC COMPLETE")
print("Results:", out)

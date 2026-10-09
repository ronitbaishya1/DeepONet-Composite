
# Windows / Abaqus Python 2025
from __future__ import print_function

import csv
import json
import os
import shutil

from odbAccess import openOdb


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

ROM_DIR = os.path.join(
    BASE_DIR, "online_results_7region_rom"
)
FINAL_DIR = os.path.join(ROM_DIR, "final")
OUT_DIR = os.path.join(ROM_DIR, "validation_fields")

DIRECT_DIR = os.path.join(
    BASE_DIR,
    "online_results_7region_direct",
    "validation_fields",
)

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)


def find_instance(odb, name):
    for key, value in odb.rootAssembly.instances.items():
        if key.lower() == name.lower():
            return value

    raise RuntimeError(
        "Cannot find Abaqus instance: " + name
    )


def extract_patch(patch):
    path = os.path.join(
        FINAL_DIR, patch + ".odb"
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    odb = openOdb(path, readOnly=True)

    try:
        instance = find_instance(
            odb, "Composite-1"
        )

        frame = odb.steps["Loading"].frames[-1]

        nodes = {
            int(n.label): tuple(
                float(v) for v in n.coordinates
            )
            for n in instance.nodes
        }

        u = {
            int(v.nodeLabel): tuple(
                float(x) for x in v.data
            )
            for v in frame.fieldOutputs["U"]
            .getSubset(region=instance).values
        }

        rf = {
            int(v.nodeLabel): tuple(
                float(x) for x in v.data
            )
            for v in frame.fieldOutputs["RF"]
            .getSubset(region=instance).values
        }

        node_csv = os.path.join(
            OUT_DIR, patch + "_nodes.csv"
        )

        with open(node_csv, "w", newline="") as f:
            w = csv.writer(f)

            w.writerow([
                "NodeLabel", "X", "Y", "Z",
                "U1", "U2", "U3",
                "RF1", "RF2", "RF3",
            ])

            for label in sorted(nodes):
                if label not in u:
                    raise RuntimeError(
                        "Missing displacement for node {}"
                        .format(label)
                    )

                w.writerow(
                    [label]
                    + list(nodes[label])
                    + list(u[label])
                    + list(
                        rf.get(
                            label,
                            (0.0, 0.0, 0.0)
                        )
                    )
                )

        # C3D8R: one integration point per element.
        s_values = (
            frame.fieldOutputs["S"]
            .getSubset(region=instance).values
        )

        le_values = (
            frame.fieldOutputs["LE"]
            .getSubset(region=instance).values
        )

        def per_element(values, field_name):
            result = {}

            for value in values:
                label = int(value.elementLabel)

                if label in result:
                    raise RuntimeError(
                        "Multiple {} integration points "
                        "in element {}".format(
                            field_name, label
                        )
                    )

                component = tuple(
                    float(x) for x in value.data
                )

                if len(component) != 6:
                    raise RuntimeError(
                        "Expected six components in {}"
                        .format(field_name)
                    )

                result[label] = component

            return result

        stresses = per_element(s_values, "S")
        strains = per_element(le_values, "LE")

        ip_csv = os.path.join(
            OUT_DIR, patch + "_ip.csv"
        )

        with open(ip_csv, "w", newline="") as f:
            w = csv.writer(f)

            w.writerow([
                "ElementLabel", "X", "Y", "Z",
                "LE11", "LE22", "LE33",
                "LE12", "LE13", "LE23",
                "S11", "S22", "S33",
                "S12", "S13", "S23",
            ])

            for element in sorted(
                instance.elements,
                key=lambda e: e.label,
            ):
                label = int(element.label)

                if (
                    label not in stresses
                    or label not in strains
                ):
                    raise RuntimeError(
                        "S/LE missing for element {}"
                        .format(label)
                    )

                xyz = [
                    nodes[int(n)]
                    for n in element.connectivity
                ]

                centroid = [
                    sum(
                        point[j] for point in xyz
                    ) / len(xyz)
                    for j in range(3)
                ]

                w.writerow(
                    [label]
                    + centroid
                    + list(strains[label])
                    + list(stresses[label])
                )

        reaction = None

        if patch == "center":
            nose = find_instance(
                odb, "Point-2"
            )

            values = (
                frame.fieldOutputs["RF"]
                .getSubset(region=nose).values
            )

            if not values:
                raise RuntimeError(
                    "Point-2 RF missing in center ODB"
                )

            reaction = sum(
                float(v.data[1])
                for v in values
            )

        print(
            "{}: {} nodes, {} elements".format(
                patch,
                len(nodes),
                len(stresses),
            )
        )

        return reaction

    finally:
        odb.close()


if __name__ == "__main__":
    # Reuse the identical full-beam FEM reference
    # that was validated in Step 64U.
    reference_files = [
        "full_reference_nodes.csv",
        "full_reference_ip.csv",
        "reaction_summary.json",
    ]

    for name in reference_files:
        source = os.path.join(
            DIRECT_DIR, name
        )

        if not os.path.isfile(source):
            raise FileNotFoundError(
                "Existing 64U reference required: "
                + source
            )

    nose_rf2 = None

    for patch_name in (
        "left", "center", "right"
    ):
        rf_value = extract_patch(patch_name)

        if patch_name == "center":
            nose_rf2 = float(rf_value)

    with open(
        os.path.join(
            DIRECT_DIR, "reaction_summary.json"
        ),
        "r",
    ) as f:
        previous_reaction = json.load(f)

    full_rf2 = float(
        previous_reaction[
            "full_FEM_nose_RF2_N"
        ]
    )

    error_N = abs(nose_rf2 - full_rf2)

    result = {
        "full_FEM_nose_RF2_N": full_rf2,
        "hybrid_center_nose_RF2_N": nose_rf2,
        "absolute_reaction_error_N": error_N,
        "relative_reaction_error":
            error_N / max(abs(full_rf2), 1e-14),
        "relative_reaction_error_percent":
            100.0 * error_N
            / max(abs(full_rf2), 1e-14),
        "reference":
            "Unchanged full-FEM reference "
            "from original 64U baseline run",
    }

    for name in (
        "full_reference_nodes.csv",
        "full_reference_ip.csv",
    ):
        shutil.copy2(
            os.path.join(DIRECT_DIR, name),
            os.path.join(OUT_DIR, name),
        )

    with open(
        os.path.join(
            OUT_DIR, "reaction_summary.json"
        ),
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    print("STEP 65I WINDOWS EXTRACTION COMPLETE")
    print(json.dumps(result, indent=2))
    print(
        "Copy rom_solution_7region.npz and "
        "validation_fields/ to the Mac."
    )

import os
import json
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--nodes_per_region",
    type=int,
    default=2,
)


parser.add_argument(
    "--elements_per_region",
    type=int,
    default=2,
)


parser.add_argument(
    "--seed",
    type=int,
    default=6420,
)


args = parser.parse_args()


# ============================================================
# PATHS
# ============================================================

VALIDATION_RESULTS_DIR = os.path.join(
    "results",
    "hybrid_online_7region_direct_validation",
)


ONLINE_VALIDATION_DIR = os.path.join(
    "data",
    "hybrid_online_7region_direct",
    "validation_fields",
)


NODE_FILE = os.path.join(
    VALIDATION_RESULTS_DIR,
    "assembled_hybrid_nodes.csv",
)


IP_FILE = os.path.join(
    VALIDATION_RESULTS_DIR,
    "assembled_hybrid_ip.csv",
)


REACTION_FILE = os.path.join(
    ONLINE_VALIDATION_DIR,
    "reaction_summary.json",
)


OUTPUT_DIR = os.path.join(
    "results",
    "fem_vs_hybrid_pointwise_comparison",
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# REGION ORDER
# ============================================================

REGION_ORDER = [

    "NO_OL",
    "FE_L",
    "NO_L",
    "FE_C",
    "NO_R",
    "FE_R",
    "NO_OR",
]


# ============================================================
# RANDOM NUMBER GENERATOR
#
# Fixed seed means the table is reproducible.
# ============================================================

rng = np.random.default_rng(
    args.seed
)


# ============================================================
# CHECK INPUTS
# ============================================================

for filename in [
    NODE_FILE,
    IP_FILE,
    REACTION_FILE,
]:

    if not os.path.isfile(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


# ============================================================
# LOAD
# ============================================================

nodes = pd.read_csv(
    NODE_FILE
)


elements = pd.read_csv(
    IP_FILE
)


with open(
    REACTION_FILE,
    "r",
) as file_object:

    reaction_data = json.load(
        file_object
    )


# ============================================================
# CHECK OWNERS
# ============================================================

for region in REGION_ORDER:

    if region not in set(
        nodes[
            "Owner"
        ]
    ):

        raise RuntimeError(
            "Node region missing: {}".format(
                region
            )
        )


    if region not in set(
        elements[
            "Owner"
        ]
    ):

        raise RuntimeError(
            "Element region missing: {}".format(
                region
            )
        )


# ============================================================
# STRATIFIED RANDOM SAMPLE
#
# This is better than taking 14 totally random points because
# otherwise the larger regions could dominate the table.
#
# We deliberately sample the same number from every region.
# ============================================================

def stratified_random_sample(
    dataframe,
    samples_per_region,
):

    selected = []


    for region in REGION_ORDER:

        region_data = dataframe[
            dataframe[
                "Owner"
            ]
            ==
            region
        ]


        if len(
            region_data
        ) == 0:

            raise RuntimeError(
                "No data for region {}".format(
                    region
                )
            )


        number_to_select = min(
            samples_per_region,
            len(
                region_data
            ),
        )


        selected_indices = rng.choice(

            region_data.index.to_numpy(),

            size=
                number_to_select,

            replace=
                False,
        )


        current = region_data.loc[
            selected_indices
        ].copy()


        current[
            "SampleNumberInRegion"
        ] = np.arange(
            1,
            len(
                current
            )
            +
            1,
        )


        selected.append(
            current
        )


    result = pd.concat(
        selected,
        ignore_index=True,
    )


    result[
        "Owner"
    ] = pd.Categorical(

        result[
            "Owner"
        ],

        categories=
            REGION_ORDER,

        ordered=
            True,
    )


    result = result.sort_values(
        [
            "Owner",
            "SampleNumberInRegion",
        ]
    ).reset_index(
        drop=True
    )


    result[
        "Owner"
    ] = result[
        "Owner"
    ].astype(
        str
    )


    return result


# ============================================================
# RANDOM NODE / ELEMENT LOCATIONS
# ============================================================

selected_nodes = stratified_random_sample(

    nodes,

    args.nodes_per_region,
)


selected_elements = stratified_random_sample(

    elements,

    args.elements_per_region,
)


# ============================================================
# RELATIVE ERROR
#
# We do not calculate percentage error when the reference
# value is essentially zero because percentage becomes
# meaningless.
# ============================================================

def relative_error_percent(
    hybrid,
    fem,
    tolerance,
):

    hybrid = np.asarray(
        hybrid,
        dtype=np.float64,
    )


    fem = np.asarray(
        fem,
        dtype=np.float64,
    )


    result = np.full(
        fem.shape,
        np.nan,
        dtype=np.float64,
    )


    valid = (
        np.abs(
            fem
        )
        >
        tolerance
    )


    result[
        valid
    ] = (

        100.0

        *

        np.abs(
            hybrid[
                valid
            ]
            -
            fem[
                valid
            ]
        )

        /

        np.abs(
            fem[
                valid
            ]
        )
    )


    return result


# ============================================================
# DISPLACEMENT TABLE
# ============================================================

displacement_table = selected_nodes[
    [
        "NodeLabel",
        "Owner",
        "X",
        "Y",
        "Z",
    ]
].copy()


for component in [
    "U1",
    "U2",
    "U3",
]:

    fem_column = (
        "FEM_{}".format(
            component
        )
    )


    hybrid_column = (
        "Hybrid_{}".format(
            component
        )
    )


    fem_values = selected_nodes[
        fem_column
    ].to_numpy(
        dtype=np.float64
    )


    hybrid_values = selected_nodes[
        hybrid_column
    ].to_numpy(
        dtype=np.float64
    )


    signed_error = (
        hybrid_values
        -
        fem_values
    )


    absolute_error = np.abs(
        signed_error
    )


    relative_error = relative_error_percent(

        hybrid_values,

        fem_values,

        tolerance=
            1.0e-10,
    )


    displacement_table[
        "FEM_{}_mm".format(
            component
        )
    ] = fem_values


    displacement_table[
        "Hybrid_{}_mm".format(
            component
        )
    ] = hybrid_values


    displacement_table[
        "SignedError_{}_mm".format(
            component
        )
    ] = signed_error


    displacement_table[
        "AbsError_{}_mm".format(
            component
        )
    ] = absolute_error


    displacement_table[
        "RelativeError_{}_percent".format(
            component
        )
    ] = relative_error


# ============================================================
# STRAIN TABLE
# ============================================================

strain_table = selected_elements[
    [
        "ElementLabel",
        "Owner",
        "X",
        "Y",
        "Z",
    ]
].copy()


STRAIN_COMPONENTS = [

    "LE11",
    "LE22",
    "LE33",
    "LE12",
    "LE13",
    "LE23",
]


for component in STRAIN_COMPONENTS:

    fem_column = (
        "FEM_{}".format(
            component
        )
    )


    hybrid_column = (
        "Hybrid_{}".format(
            component
        )
    )


    fem_values = selected_elements[
        fem_column
    ].to_numpy(
        dtype=np.float64
    )


    hybrid_values = selected_elements[
        hybrid_column
    ].to_numpy(
        dtype=np.float64
    )


    signed_error = (
        hybrid_values
        -
        fem_values
    )


    absolute_error = np.abs(
        signed_error
    )


    relative_error = relative_error_percent(

        hybrid_values,

        fem_values,

        tolerance=
            1.0e-12,
    )


    strain_table[
        "FEM_{}".format(
            component
        )
    ] = fem_values


    strain_table[
        "Hybrid_{}".format(
            component
        )
    ] = hybrid_values


    strain_table[
        "SignedError_{}".format(
            component
        )
    ] = signed_error


    strain_table[
        "AbsError_{}".format(
            component
        )
    ] = absolute_error


    strain_table[
        "RelativeError_{}_percent".format(
            component
        )
    ] = relative_error


# ============================================================
# STRESS TABLE
# ============================================================

stress_table = selected_elements[
    [
        "ElementLabel",
        "Owner",
        "X",
        "Y",
        "Z",
    ]
].copy()


STRESS_COMPONENTS = [

    "S11",
    "S22",
    "S33",
    "S12",
    "S13",
    "S23",
]


for component in STRESS_COMPONENTS:

    fem_column = (
        "FEM_{}".format(
            component
        )
    )


    hybrid_column = (
        "Hybrid_{}".format(
            component
        )
    )


    fem_values = selected_elements[
        fem_column
    ].to_numpy(
        dtype=np.float64
    )


    hybrid_values = selected_elements[
        hybrid_column
    ].to_numpy(
        dtype=np.float64
    )


    signed_error = (
        hybrid_values
        -
        fem_values
    )


    absolute_error = np.abs(
        signed_error
    )


    relative_error = relative_error_percent(

        hybrid_values,

        fem_values,

        tolerance=
            1.0e-8,
    )


    stress_table[
        "FEM_{}_MPa".format(
            component
        )
    ] = fem_values


    stress_table[
        "Hybrid_{}_MPa".format(
            component
        )
    ] = hybrid_values


    stress_table[
        "SignedError_{}_MPa".format(
            component
        )
    ] = signed_error


    stress_table[
        "AbsError_{}_MPa".format(
            component
        )
    ] = absolute_error


    stress_table[
        "RelativeError_{}_percent".format(
            component
        )
    ] = relative_error


# ============================================================
# REACTION TABLE
# ============================================================

full_fem_reaction = float(
    reaction_data[
        "full_FEM_nose_RF2_N"
    ]
)


hybrid_reaction = float(
    reaction_data[
        "hybrid_center_nose_RF2_N"
    ]
)


reaction_signed_error = (
    hybrid_reaction
    -
    full_fem_reaction
)


reaction_absolute_error = abs(
    reaction_signed_error
)


reaction_relative_error = (

    100.0

    *

    reaction_absolute_error

    /

    (
        abs(
            full_fem_reaction
        )
        +
        1.0e-14
    )
)


reaction_table = pd.DataFrame(
    [
        {
            "Quantity":
                "Nose reaction RF2",

            "Unit":
                "N",

            "Full_FEM":
                full_fem_reaction,

            "Hybrid":
                hybrid_reaction,

            "SignedError":
                reaction_signed_error,

            "AbsError":
                reaction_absolute_error,

            "RelativeError_percent":
                reaction_relative_error,
        }
    ]
)


# ============================================================
# SELECTED LOCATION TABLE
#
# Useful so we always know exactly which random locations were
# used in the comparison.
# ============================================================

selected_node_locations = displacement_table[
    [
        "NodeLabel",
        "Owner",
        "X",
        "Y",
        "Z",
    ]
].copy()


selected_element_locations = strain_table[
    [
        "ElementLabel",
        "Owner",
        "X",
        "Y",
        "Z",
    ]
].copy()


# ============================================================
# SAVE CSV FILES
# ============================================================

displacement_csv = os.path.join(
    OUTPUT_DIR,
    "01_displacement_value_comparison.csv",
)


strain_csv = os.path.join(
    OUTPUT_DIR,
    "02_strain_value_comparison.csv",
)


stress_csv = os.path.join(
    OUTPUT_DIR,
    "03_stress_value_comparison.csv",
)


reaction_csv = os.path.join(
    OUTPUT_DIR,
    "04_reaction_value_comparison.csv",
)


displacement_table.to_csv(
    displacement_csv,
    index=False,
)


strain_table.to_csv(
    strain_csv,
    index=False,
)


stress_table.to_csv(
    stress_csv,
    index=False,
)


reaction_table.to_csv(
    reaction_csv,
    index=False,
)


# ============================================================
# LONG-FORM PAPER-FRIENDLY TABLE
#
# Each row is:
#
# Region | location | quantity | FEM | Hybrid | error
#
# This is easier to use directly in a manuscript than the
# very wide tables.
# ============================================================

long_rows = []


# ============================================================
# DISPLACEMENT LONG FORM
# ============================================================

for _, row in selected_nodes.iterrows():

    for component in [
        "U1",
        "U2",
        "U3",
    ]:

        fem = float(
            row[
                "FEM_{}".format(
                    component
                )
            ]
        )


        hybrid = float(
            row[
                "Hybrid_{}".format(
                    component
                )
            ]
        )


        absolute_error = abs(
            hybrid
            -
            fem
        )


        if abs(
            fem
        ) > 1.0e-10:

            relative_error = (

                100.0
                *
                absolute_error
                /
                abs(
                    fem
                )
            )

        else:

            relative_error = np.nan


        long_rows.append(
            {
                "LocationType":
                    "Node",

                "Label":
                    int(
                        row[
                            "NodeLabel"
                        ]
                    ),

                "Region":
                    row[
                        "Owner"
                    ],

                "X":
                    row[
                        "X"
                    ],

                "Y":
                    row[
                        "Y"
                    ],

                "Z":
                    row[
                        "Z"
                    ],

                "Field":
                    "Displacement",

                "Component":
                    component,

                "Unit":
                    "mm",

                "Full_FEM":
                    fem,

                "Hybrid":
                    hybrid,

                "SignedError":
                    hybrid
                    -
                    fem,

                "AbsError":
                    absolute_error,

                "RelativeError_percent":
                    relative_error,
            }
        )


# ============================================================
# STRAIN LONG FORM
# ============================================================

for _, row in selected_elements.iterrows():

    for component in STRAIN_COMPONENTS:

        fem = float(
            row[
                "FEM_{}".format(
                    component
                )
            ]
        )


        hybrid = float(
            row[
                "Hybrid_{}".format(
                    component
                )
            ]
        )


        absolute_error = abs(
            hybrid
            -
            fem
        )


        if abs(
            fem
        ) > 1.0e-12:

            relative_error = (

                100.0
                *
                absolute_error
                /
                abs(
                    fem
                )
            )

        else:

            relative_error = np.nan


        long_rows.append(
            {
                "LocationType":
                    "Element",

                "Label":
                    int(
                        row[
                            "ElementLabel"
                        ]
                    ),

                "Region":
                    row[
                        "Owner"
                    ],

                "X":
                    row[
                        "X"
                    ],

                "Y":
                    row[
                        "Y"
                    ],

                "Z":
                    row[
                        "Z"
                    ],

                "Field":
                    "Logarithmic strain",

                "Component":
                    component,

                "Unit":
                    "dimensionless",

                "Full_FEM":
                    fem,

                "Hybrid":
                    hybrid,

                "SignedError":
                    hybrid
                    -
                    fem,

                "AbsError":
                    absolute_error,

                "RelativeError_percent":
                    relative_error,
            }
        )


# ============================================================
# STRESS LONG FORM
# ============================================================

for _, row in selected_elements.iterrows():

    for component in STRESS_COMPONENTS:

        fem = float(
            row[
                "FEM_{}".format(
                    component
                )
            ]
        )


        hybrid = float(
            row[
                "Hybrid_{}".format(
                    component
                )
            ]
        )


        absolute_error = abs(
            hybrid
            -
            fem
        )


        if abs(
            fem
        ) > 1.0e-8:

            relative_error = (

                100.0
                *
                absolute_error
                /
                abs(
                    fem
                )
            )

        else:

            relative_error = np.nan


        long_rows.append(
            {
                "LocationType":
                    "Element",

                "Label":
                    int(
                        row[
                            "ElementLabel"
                        ]
                    ),

                "Region":
                    row[
                        "Owner"
                    ],

                "X":
                    row[
                        "X"
                    ],

                "Y":
                    row[
                        "Y"
                    ],

                "Z":
                    row[
                        "Z"
                    ],

                "Field":
                    "Stress",

                "Component":
                    component,

                "Unit":
                    "MPa",

                "Full_FEM":
                    fem,

                "Hybrid":
                    hybrid,

                "SignedError":
                    hybrid
                    -
                    fem,

                "AbsError":
                    absolute_error,

                "RelativeError_percent":
                    relative_error,
            }
        )


long_table = pd.DataFrame(
    long_rows
)


long_table.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "05_all_pointwise_values_long_format.csv",
    ),

    index=False,
)


# ============================================================
# EXCEL WORKBOOK
# ============================================================

excel_file = os.path.join(
    OUTPUT_DIR,
    "FEM_vs_Hybrid_40FE_60NO_value_comparison.xlsx",
)


excel_created = False


try:

    from openpyxl.styles import (
        Font,
        Alignment,
    )


    from openpyxl.utils import (
        get_column_letter,
    )


    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:

        displacement_table.to_excel(
            writer,
            sheet_name="Displacement",
            index=False,
        )


        strain_table.to_excel(
            writer,
            sheet_name="Strain",
            index=False,
        )


        stress_table.to_excel(
            writer,
            sheet_name="Stress",
            index=False,
        )


        reaction_table.to_excel(
            writer,
            sheet_name="Reaction",
            index=False,
        )


        long_table.to_excel(
            writer,
            sheet_name="Long Format",
            index=False,
        )


        # ====================================================
        # FORMAT WORKBOOK
        # ====================================================

        workbook = writer.book


        for worksheet in workbook.worksheets:

            worksheet.freeze_panes = "A2"


            for cell in worksheet[
                1
            ]:

                cell.font = Font(
                    bold=True
                )


                cell.alignment = Alignment(
                    horizontal="center",
                    vertical="center",
                )


            for column_cells in worksheet.columns:

                maximum_length = 0


                column_letter = get_column_letter(
                    column_cells[
                        0
                    ].column
                )


                for cell in column_cells:

                    value = (
                        ""
                        if cell.value is None
                        else str(
                            cell.value
                        )
                    )


                    maximum_length = max(
                        maximum_length,
                        len(
                            value
                        ),
                    )


                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    maximum_length
                    +
                    2,
                    24,
                )


    excel_created = True


except Exception as error:

    print("")
    print(
        "Excel workbook was not created."
    )


    print(
        "CSV files were still created correctly."
    )


    print(
        "Excel error:",
        error
    )


# ============================================================
# METADATA
# ============================================================

metadata = {

    "RandomSeed":
        args.seed,

    "NodesPerRegion":
        args.nodes_per_region,

    "ElementsPerRegion":
        args.elements_per_region,

    "Regions":
        REGION_ORDER,

    "TotalRandomNodes":
        int(
            len(
                selected_nodes
            )
        ),

    "TotalRandomElements":
        int(
            len(
                selected_elements
            )
        ),

    "NodeSource":
        NODE_FILE,

    "ElementSource":
        IP_FILE,

    "ReactionSource":
        REACTION_FILE,

    "ExcelCreated":
        excel_created,
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "comparison_metadata.json",
    ),
    "w",
) as file_object:

    json.dump(
        metadata,
        file_object,
        indent=4,
    )


# ============================================================
# SIMPLE TEXT REPORT
# ============================================================

text_file = os.path.join(
    OUTPUT_DIR,
    "FEM_vs_Hybrid_value_comparison_report.txt",
)


with open(
    text_file,
    "w",
) as file_object:

    file_object.write(
        "FEM VS HYBRID VALUE COMPARISON\n"
    )


    file_object.write(
        "==============================\n\n"
    )


    file_object.write(
        "Random seed: {}\n".format(
            args.seed
        )
    )


    file_object.write(
        "Nodes per region: {}\n".format(
            args.nodes_per_region
        )
    )


    file_object.write(
        "Elements per region: {}\n\n".format(
            args.elements_per_region
        )
    )


    file_object.write(
        "DISPLACEMENT VALUES\n"
    )


    file_object.write(
        "-------------------\n"
    )


    file_object.write(
        displacement_table.to_string(
            index=False
        )
    )


    file_object.write(
        "\n\n"
    )


    file_object.write(
        "REACTION\n"
    )


    file_object.write(
        "--------\n"
    )


    file_object.write(
        reaction_table.to_string(
            index=False
        )
    )


# ============================================================
# PRINT
# ============================================================

print("")
print(
    "=========================================="
)

print(
    "STEP 64W2 COMPLETE"
)

print(
    "=========================================="
)


print("")
print(
    "Random seed:",
    args.seed
)


print(
    "Random nodes:",
    len(
        selected_nodes
    )
)


print(
    "Random elements:",
    len(
        selected_elements
    )
)


print("")
print(
    "REACTION COMPARISON"
)


print(
    reaction_table.to_string(
        index=False
    )
)


print("")
print(
    "Files saved to:"
)


print(
    OUTPUT_DIR
)


if excel_created:

    print("")
    print(
        "Excel workbook:"
    )


    print(
        excel_file
    )
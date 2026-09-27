import os

import numpy as np

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data"


RAW_DIR = os.path.join(
    DATA_DIR,
    "raw_full_tensor",
)


# ============================================================
# COMPONENT ORDER
# ============================================================

STRESS_COMPONENTS = [

    "S11",
    "S22",
    "S33",

    "S12",
    "S13",
    "S23",

]


STRAIN_COMPONENTS = [

    "LE11",
    "LE22",
    "LE33",

    "LE12",
    "LE13",
    "LE23",

]


# ============================================================
# CASE ORDER
# ============================================================

case_table = pd.read_csv(

    os.path.join(
        DATA_DIR,
        "case_ids.csv",
    )

)


case_table = case_table.sort_values(
    "Index"
)


case_ids = (
    case_table[
        "CaseID"
    ]
    .astype(str)
    .tolist()
)


# ============================================================
# STORAGE
# ============================================================

stress_all = []

strain_all = []


reference_keys = None

reference_coordinates = None


# ============================================================
# LOOP
# ============================================================

for (
    case_counter,
    case_id,
) in enumerate(
    case_ids,
    start=1,
):

    filename = os.path.join(

        RAW_DIR,

        case_id,

        "full_tensor.csv",

    )


    if not os.path.exists(
        filename
    ):

        raise FileNotFoundError(
            filename
        )


    data = pd.read_csv(
        filename
    )


    data = data.sort_values(

        [

            "ElementLabel",

            "IntegrationPoint",

        ]

    ).reset_index(
        drop=True
    )


    keys = data[
        [
            "ElementLabel",
            "IntegrationPoint",
        ]
    ].to_numpy()


    coordinates = data[
        [
            "X_mm",
            "Y_mm",
            "Z_mm",
        ]
    ].to_numpy(
        dtype=np.float32
    )


    if (
        reference_keys
        is None
    ):

        reference_keys = (
            keys.copy()
        )

        reference_coordinates = (
            coordinates.copy()
        )


        print(
            "Integration-point count:",
            len(
                reference_keys
            ),
        )


    else:

        if not np.array_equal(
            keys,
            reference_keys,
        ):

            raise RuntimeError(

                "{} has different "
                "element/IP ordering."

                .format(
                    case_id
                )

            )


        if not np.allclose(

            coordinates,

            reference_coordinates,

            rtol=0.0,

            atol=1.0e-7,

        ):

            raise RuntimeError(

                "{} has different "
                "integration-point "
                "coordinates."

                .format(
                    case_id
                )

            )


    stress_all.append(

        data[
            STRESS_COMPONENTS
        ].to_numpy(
            dtype=np.float32
        )

    )


    strain_all.append(

        data[
            STRAIN_COMPONENTS
        ].to_numpy(
            dtype=np.float32
        )

    )


    print(

        "Loaded {:2d}/{}: {}"

        .format(

            case_counter,

            len(
                case_ids
            ),

            case_id,

        )

    )


# ============================================================
# STACK
# ============================================================

stress_all = np.stack(
    stress_all,
    axis=0,
)


strain_all = np.stack(
    strain_all,
    axis=0,
)


# ============================================================
# SAVE
# ============================================================

np.save(

    os.path.join(
        DATA_DIR,
        "ip_coordinates.npy",
    ),

    reference_coordinates,

)


np.save(

    os.path.join(
        DATA_DIR,
        "ip_keys.npy",
    ),

    reference_keys,

)


np.save(

    os.path.join(
        DATA_DIR,
        "S_tensor.npy",
    ),

    stress_all,

)


np.save(

    os.path.join(
        DATA_DIR,
        "LE_tensor.npy",
    ),

    strain_all,

)


print("")
print(
    "Assembly complete."
)


print(
    "ip_coordinates:",
    reference_coordinates.shape,
)


print(
    "S_tensor:",
    stress_all.shape,
)


print(
    "LE_tensor:",
    strain_all.shape,
)


print("")
print(
    "Tensor order:"
)


print(
    "[11,22,33,12,13,23]"
)
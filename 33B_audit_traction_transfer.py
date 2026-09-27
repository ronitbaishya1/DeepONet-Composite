import os
import argparse

import numpy as np
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--segment",
    choices=[
        "left",
        "right",
    ],
    required=True,
)


parser.add_argument(
    "--dataset_dir",
    required=True,
)


parser.add_argument(
    "--extracted_dir",
    required=True,
)


parser.add_argument(
    "--interface_dir",
    default=
        "data/hybrid_interfaces_trainonly",
)


parser.add_argument(
    "--output_dir",
    default=
        "results/interface_information_audit",
)


args = parser.parse_args()


os.makedirs(
    args.output_dir,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

if args.segment == "left":

    interface_names = [
        "m6",
        "m2",
    ]

else:

    interface_names = [
        "p2",
        "p6",
    ]


# ============================================================
# HELPERS
# ============================================================

def coordinate_key(
    x,
    y,
    z,
):

    return (
        round(float(x), 6),
        round(float(y), 6),
        round(float(z), 6),
    )


def identify_split_column(
    dataframe,
):

    for candidate in [
        "Split",
        "split",
    ]:

        if candidate in dataframe.columns:

            return candidate


    raise RuntimeError(
        "Split column not found."
    )


def split_indices(
    dataframe,
    split_name,
):

    column = identify_split_column(
        dataframe
    )


    values = (
        dataframe[
            column
        ]
        .astype(str)
        .str.lower()
    )


    if split_name == "validation":

        mask = values.isin(
            [
                "validation",
                "val",
            ]
        )

    else:

        mask = (
            values
            ==
            split_name.lower()
        )


    if "Index" in dataframe.columns:

        return dataframe.loc[
            mask,
            "Index",
        ].to_numpy(
            dtype=int
        )


    return np.where(
        mask.to_numpy()
    )[0]


def relative_l2(
    prediction,
    truth,
):

    return (
        np.linalg.norm(
            prediction
            -
            truth
        )
        /
        (
            np.linalg.norm(
                truth
            )
            +
            1.0e-14
        )
    )


def fit_pca(
    matrix,
    training_indices,
):

    training = matrix[
        training_indices
    ]


    mean = training.mean(
        axis=0
    )


    centered = (
        training
        -
        mean
    )


    _, singular_values, Vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )


    basis = Vt.T


    energy = (
        singular_values
        ** 2
    )


    cumulative_energy = (
        np.cumsum(
            energy
        )
        /
        np.sum(
            energy
        )
    )


    return (
        mean,
        basis,
        cumulative_energy,
    )


# ============================================================
# SPLITS / CASE IDS
# ============================================================

split_dataframe = pd.read_csv(
    os.path.join(
        args.dataset_dir,
        "split_assignment.csv",
    )
)


if "CaseID" not in split_dataframe.columns:

    raise RuntimeError(
        "split_assignment.csv must contain CaseID."
    )


case_ids = (
    split_dataframe[
        "CaseID"
    ]
    .astype(str)
    .to_list()
)


train_indices = split_indices(
    split_dataframe,
    "train",
)


validation_indices = split_indices(
    split_dataframe,
    "validation",
)


test_indices = split_indices(
    split_dataframe,
    "test",
)


# ============================================================
# PROCESS INTERFACES
# ============================================================

capture_rows = []

traction_pca_rows = []


for interface_name in interface_names:

    interface = np.load(
        os.path.join(
            args.interface_dir,
            "interface_{}.npz".format(
                interface_name
            ),
        )
    )


    interface_coordinates = interface[
        "coordinates"
    ].astype(
        np.float64
    )


    displacement_basis = interface[
        "basis"
    ].astype(
        np.float64
    )


    reaction_matrix = []


    for case_id in case_ids:

        node_file = os.path.join(
            args.extracted_dir,
            "{}_nodes.csv".format(
                case_id
            ),
        )


        if not os.path.isfile(
            node_file
        ):

            raise FileNotFoundError(
                node_file
            )


        nodes = pd.read_csv(
            node_file
        )


        reaction_map = {}


        for _, row in nodes.iterrows():

            key = coordinate_key(
                row[
                    "X"
                ],
                row[
                    "Y"
                ],
                row[
                    "Z"
                ],
            )


            reaction_map[
                key
            ] = np.array(
                [
                    row[
                        "RF1"
                    ],
                    row[
                        "RF2"
                    ],
                    row[
                        "RF3"
                    ],
                ],
                dtype=np.float64,
            )


        interface_reaction = []


        for xyz in interface_coordinates:

            key = coordinate_key(
                xyz[0],
                xyz[1],
                xyz[2],
            )


            if key not in reaction_map:

                raise RuntimeError(
                    "Interface coordinate {} "
                    "not found in {}".format(
                        key,
                        node_file,
                    )
                )


            interface_reaction.append(
                reaction_map[
                    key
                ]
            )


        interface_reaction = np.asarray(
            interface_reaction,
            dtype=np.float64,
        ).reshape(
            -1
        )


        reaction_matrix.append(
            interface_reaction
        )


    reaction_matrix = np.stack(
        reaction_matrix,
        axis=0,
    )


    # --------------------------------------------------------
    # HOW MUCH FORCE FIELD LIES IN DISPLACEMENT PCA SUBSPACE?
    # --------------------------------------------------------

    projector = (
        displacement_basis
        @
        displacement_basis.T
    )


    captures = []


    for reaction in reaction_matrix:

        projected = (
            projector
            @
            reaction
        )


        captures.append(
            (
                np.linalg.norm(
                    projected
                )
                ** 2
            )
            /
            (
                np.linalg.norm(
                    reaction
                )
                ** 2
                +
                1.0e-20
            )
        )


    captures = np.asarray(
        captures
    )


    capture_rows.append(
        {

            "Segment":
                args.segment,

            "Interface":
                interface_name,

            "DisplacementModes":
                displacement_basis.shape[1],

            "MeanTractionEnergyCaptured":
                float(
                    np.mean(
                        captures
                    )
                ),

            "MedianTractionEnergyCaptured":
                float(
                    np.median(
                        captures
                    )
                ),

            "MinTractionEnergyCaptured":
                float(
                    np.min(
                        captures
                    )
                ),

            "MeanTractionEnergyCaptured_percent":
                float(
                    100.0
                    *
                    np.mean(
                        captures
                    )
                ),

            "MinTractionEnergyCaptured_percent":
                float(
                    100.0
                    *
                    np.min(
                        captures
                    )
                ),
        }
    )


    # --------------------------------------------------------
    # TRACTION-SPECIFIC PCA
    # --------------------------------------------------------

    (
        traction_mean,
        traction_basis,
        traction_energy,
    ) = fit_pca(
        reaction_matrix,
        train_indices,
    )


    candidates = [
        2,
        4,
        5,
        6,
        8,
        10,
        12,
        16,
        20,
    ]


    candidates = [

        modes

        for modes in candidates

        if modes <= traction_basis.shape[1]
    ]


    for modes in candidates:

        basis = traction_basis[
            :,
            :modes
        ]


        for split_name, indices in [

            (
                "train",
                train_indices,
            ),

            (
                "validation",
                validation_indices,
            ),

            (
                "test",
                test_indices,
            ),
        ]:

            errors = []


            for case_index in indices:

                vector = reaction_matrix[
                    case_index
                ]


                coefficients = (
                    basis.T
                    @
                    (
                        vector
                        -
                        traction_mean
                    )
                )


                reconstructed = (
                    traction_mean
                    +
                    basis
                    @
                    coefficients
                )


                errors.append(
                    relative_l2(
                        reconstructed,
                        vector,
                    )
                )


            errors = np.asarray(
                errors
            )


            traction_pca_rows.append(
                {

                    "Segment":
                        args.segment,

                    "Interface":
                        interface_name,

                    "Modes":
                        modes,

                    "Split":
                        split_name,

                    "CumulativeEnergy":
                        float(
                            traction_energy[
                                modes
                                -
                                1
                            ]
                        ),

                    "MeanRelativeL2_percent":
                        float(
                            100.0
                            *
                            np.mean(
                                errors
                            )
                        ),

                    "MaxRelativeL2_percent":
                        float(
                            100.0
                            *
                            np.max(
                                errors
                            )
                        ),
                }
            )


# ============================================================
# SAVE
# ============================================================

capture_dataframe = pd.DataFrame(
    capture_rows
)


traction_dataframe = pd.DataFrame(
    traction_pca_rows
)


capture_file = os.path.join(
    args.output_dir,
    "{}_traction_capture.csv".format(
        args.segment
    ),
)


traction_file = os.path.join(
    args.output_dir,
    "{}_traction_pca_audit.csv".format(
        args.segment
    ),
)


capture_dataframe.to_csv(
    capture_file,
    index=False,
)


traction_dataframe.to_csv(
    traction_file,
    index=False,
)


print("")
print(
    capture_dataframe.to_string(
        index=False
    )
)


print("")
print(
    "Saved:"
)

print(
    capture_file
)

print(
    traction_file
)
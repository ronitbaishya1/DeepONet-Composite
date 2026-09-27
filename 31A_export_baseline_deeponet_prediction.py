import os
import json
import inspect
import importlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.getcwd()

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "vector_deeponet_separate",
)

CHECKPOINT_FILE = os.path.join(
    MODEL_DIR,
    "best_separate_vector_deeponet.pt",
)

OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "baseline_deeponet_nodes.csv",
)

METRICS_FILE = os.path.join(
    MODEL_DIR,
    "baseline_deeponet_metrics.csv",
)

REFERENCE_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "hybrid_online_baseline",
    "validation_fields",
    "full_reference_nodes.csv",
)


# ============================================================
# BASELINE MATERIAL
# ============================================================

E1 = 45000.0
E2 = 12000.0
G12 = 4500.0


# ============================================================
# DATA FILES
# ============================================================

PARAMETERS_FILE = os.path.join(
    DATA_DIR,
    "parameters.npy",
)

COORDINATES_FILE = os.path.join(
    DATA_DIR,
    "coordinates.npy",
)

NODE_LABELS_FILE = os.path.join(
    DATA_DIR,
    "node_labels.npy",
)

U1_FILE = os.path.join(
    DATA_DIR,
    "U1.npy",
)

U2_FILE = os.path.join(
    DATA_DIR,
    "U2.npy",
)

U3_FILE = os.path.join(
    DATA_DIR,
    "U3.npy",
)

SPLIT_FILE = os.path.join(
    DATA_DIR,
    "split_assignment.csv",
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    PARAMETERS_FILE,
    COORDINATES_FILE,
    U1_FILE,
    U2_FILE,
    U3_FILE,
    SPLIT_FILE,
    CHECKPOINT_FILE,
]


for path in required_files:

    if not os.path.isfile(path):

        raise FileNotFoundError(
            path
        )


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

try:

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
        weights_only=False,
    )

except TypeError:

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location=DEVICE,
    )


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        tensor_values = all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )

        if tensor_values:

            state_dict = checkpoint

        else:

            raise RuntimeError(
                "Could not find model_state_dict "
                "inside checkpoint."
            )

else:

    raise RuntimeError(
        "Unexpected checkpoint format."
    )


# ============================================================
# LOAD MODEL CLASS
# ============================================================

module = importlib.import_module(
    "src.separate_vector_deeponet"
)


candidate_class_names = [

    "SeparateVectorDeepONet",
    "SeparateDeepONet",
    "VectorDeepONet",
]


model_class = None


for class_name in candidate_class_names:

    if hasattr(
        module,
        class_name,
    ):

        candidate = getattr(
            module,
            class_name,
        )

        if (
            inspect.isclass(candidate)
            and
            issubclass(
                candidate,
                nn.Module,
            )
        ):

            model_class = candidate

            break


# ============================================================
# FALLBACK:
# FIND LIKELY NN.MODULE CLASS
# ============================================================

if model_class is None:

    module_classes = []


    for name, candidate in inspect.getmembers(
        module,
        inspect.isclass,
    ):

        if (
            issubclass(
                candidate,
                nn.Module,
            )
            and
            candidate is not nn.Module
        ):

            if (
                "vector"
                in name.lower()
                or
                "separate"
                in name.lower()
            ):

                module_classes.append(
                    candidate
                )


    if len(
        module_classes
    ) == 1:

        model_class = module_classes[
            0
        ]


if model_class is None:

    raise RuntimeError(
        "Could not automatically identify the "
        "full-domain separate-vector DeepONet class "
        "inside src/separate_vector_deeponet.py."
    )


print("")
print(
    "Using model class:",
    model_class.__name__,
)


# ============================================================
# CONSTRUCTOR VALUES
# ============================================================

latent_dim = 128
hidden_dim = 128


if isinstance(
    checkpoint,
    dict,
):

    latent_dim = int(
        checkpoint.get(
            "latent_dim",
            latent_dim,
        )
    )

    hidden_dim = int(
        checkpoint.get(
            "hidden_dim",
            hidden_dim,
        )
    )


signature = inspect.signature(
    model_class.__init__
)


constructor_kwargs = {}


known_values = {

    "branch_dim":
        3,

    "branch_input_dim":
        3,

    "input_dim":
        3,

    "trunk_dim":
        3,

    "trunk_input_dim":
        3,

    "coordinate_dim":
        3,

    "coord_dim":
        3,

    "latent_dim":
        latent_dim,

    "basis_dim":
        latent_dim,

    "p":
        latent_dim,

    "hidden_dim":
        hidden_dim,

    "width":
        hidden_dim,

    "num_outputs":
        3,

    "n_outputs":
        3,

    "output_dim":
        3,
}


for parameter_name, parameter in signature.parameters.items():

    if parameter_name == "self":

        continue


    if parameter_name in known_values:

        constructor_kwargs[
            parameter_name
        ] = known_values[
            parameter_name
        ]

        continue


    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        parameter_name in checkpoint
        and
        not isinstance(
            checkpoint[
                parameter_name
            ],
            dict,
        )
    ):

        constructor_kwargs[
            parameter_name
        ] = checkpoint[
            parameter_name
        ]

        continue


    if parameter.default is not inspect._empty:

        continue


    raise RuntimeError(
        "Cannot automatically determine required "
        "constructor argument '{}'.\n"
        "Model signature is:\n{}".format(
            parameter_name,
            signature,
        )
    )


print(
    "Constructor arguments:",
    constructor_kwargs,
)


# ============================================================
# CREATE MODEL
# ============================================================

model = model_class(
    **constructor_kwargs
).to(
    DEVICE
)


# ============================================================
# REMOVE OPTIONAL 'module.' PREFIX
# ============================================================

clean_state_dict = {}


for key, value in state_dict.items():

    if key.startswith(
        "module."
    ):

        new_key = key[
            len(
                "module."
            ):
        ]

    else:

        new_key = key


    clean_state_dict[
        new_key
    ] = value


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

model.load_state_dict(
    clean_state_dict,
    strict=True,
)


model.eval()


print(
    "Checkpoint loaded successfully."
)


# ============================================================
# LOAD ORIGINAL TRAINING DATA
# ============================================================

parameters = np.load(
    PARAMETERS_FILE
).astype(
    np.float32
)


coordinates = np.load(
    COORDINATES_FILE
).astype(
    np.float32
)


U1 = np.load(
    U1_FILE
).astype(
    np.float32
)


U2 = np.load(
    U2_FILE
).astype(
    np.float32
)


U3 = np.load(
    U3_FILE
).astype(
    np.float32
)


U = np.stack(
    [
        U1,
        U2,
        U3,
    ],
    axis=-1,
)


# ============================================================
# NODE LABELS
# ============================================================

if os.path.isfile(
    NODE_LABELS_FILE
):

    node_labels = np.load(
        NODE_LABELS_FILE
    )

else:

    node_labels = np.arange(
        1,
        coordinates.shape[
            0
        ]
        +
        1,
    )


# ============================================================
# TRAIN SPLIT
# ============================================================

split_table = pd.read_csv(
    SPLIT_FILE
)


split_column = None


for candidate in [
    "Split",
    "split",
    "Set",
    "set",
    "Partition",
    "partition",
]:

    if candidate in split_table.columns:

        split_column = candidate

        break


if split_column is None:

    raise RuntimeError(
        "Could not identify train/validation/test "
        "column in split_assignment.csv."
    )


train_mask = (
    split_table[
        split_column
    ]
    .astype(
        str
    )
    .str.lower()
    ==
    "train"
)


train_indices = np.where(
    train_mask.to_numpy()
)[
    0
]


if len(
    train_indices
) == 0:

    raise RuntimeError(
        "No training cases found."
    )


print(
    "Training cases used for normalization:",
    len(
        train_indices
    ),
)


# ============================================================
# NORMALIZATION
#
# Same train-only normalization used in the project:
#
# branch: mean / std from training realizations
# trunk: coordinate min-max -> [-1,1]
# displacement: train-only component mean / std
# ============================================================

branch_mean = parameters[
    train_indices
].mean(
    axis=0
)


branch_std = parameters[
    train_indices
].std(
    axis=0
)


branch_std = np.maximum(
    branch_std,
    1.0e-8,
)


coordinate_min = coordinates.min(
    axis=0
)


coordinate_max = coordinates.max(
    axis=0
)


coordinate_range = (
    coordinate_max
    -
    coordinate_min
)


coordinate_range = np.maximum(
    coordinate_range,
    1.0e-8,
)


U_train = U[
    train_indices
]


output_mean = U_train.reshape(
    -1,
    3,
).mean(
    axis=0
)


output_std = U_train.reshape(
    -1,
    3,
).std(
    axis=0
)


output_std = np.maximum(
    output_std,
    1.0e-8,
)


# ============================================================
# BASELINE INPUT
# ============================================================

baseline_branch = np.array(
    [
        E1,
        E2,
        G12,
    ],
    dtype=np.float32,
)


baseline_branch_normalized = (
    baseline_branch
    -
    branch_mean
) / branch_std


coordinates_normalized = (

    2.0
    *
    (
        coordinates
        -
        coordinate_min
    )
    /
    coordinate_range
    -
    1.0
)


branch_tensor = torch.tensor(
    baseline_branch_normalized,
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(
    0
)


coordinate_tensor = torch.tensor(
    coordinates_normalized,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MODEL FORWARD
# ============================================================

with torch.no_grad():

    model_output = model(
        branch_tensor,
        coordinate_tensor,
    )


# ============================================================
# STANDARDIZE DIFFERENT POSSIBLE OUTPUT FORMATS
# ============================================================

def standardize_output(
    output,
):

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    if torch.is_tensor(
        output
    ):

        array = output.detach().cpu()


        if (
            array.ndim == 3
            and
            array.shape[
                0
            ] == 1
            and
            array.shape[
                -1
            ] == 3
        ):

            return array[
                0
            ].numpy()


        if (
            array.ndim == 3
            and
            array.shape[
                0
            ] == 1
            and
            array.shape[
                1
            ] == 3
        ):

            return (
                array[
                    0
                ]
                .transpose(
                    0,
                    1,
                )
                .numpy()
            )


        if (
            array.ndim == 2
            and
            array.shape[
                1
            ] == 3
        ):

            return array.numpy()


        if (
            array.ndim == 2
            and
            array.shape[
                0
            ] == 3
        ):

            return (
                array
                .transpose(
                    0,
                    1,
                )
                .numpy()
            )


    # --------------------------------------------------------
    # Tuple / list:
    # (U1,U2,U3)
    # --------------------------------------------------------

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):

        if len(
            output
        ) == 3:

            components = []


            for component in output:

                component = (
                    component
                    .detach()
                    .cpu()
                    .numpy()
                )


                component = np.squeeze(
                    component
                )


                components.append(
                    component
                )


            return np.stack(
                components,
                axis=-1,
            )


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        output,
        dict,
    ):

        keys_lower = {

            key.lower():
                key

            for key in output.keys()
        }


        component_keys = []


        for component_name in [
            "u1",
            "u2",
            "u3",
        ]:

            if component_name not in keys_lower:

                raise RuntimeError(
                    "Could not identify {} "
                    "in model output dictionary.".format(
                        component_name
                    )
                )


            component_keys.append(
                keys_lower[
                    component_name
                ]
            )


        components = []


        for key in component_keys:

            component = (
                output[
                    key
                ]
                .detach()
                .cpu()
                .numpy()
            )


            component = np.squeeze(
                component
            )


            components.append(
                component
            )


        return np.stack(
            components,
            axis=-1,
        )


    raise RuntimeError(
        "Unsupported model output format.\n"
        "Output type: {}".format(
            type(
                output
            )
        )
    )


prediction_normalized = standardize_output(
    model_output
)


if prediction_normalized.shape != (
    coordinates.shape[
        0
    ],
    3,
):

    raise RuntimeError(
        "Unexpected prediction shape: {}\n"
        "Expected: ({}, 3)".format(
            prediction_normalized.shape,
            coordinates.shape[
                0
            ],
        )
    )


# ============================================================
# DENORMALIZE
# ============================================================

prediction = (

    prediction_normalized
    *
    output_std.reshape(
        1,
        3,
    )

    +
    output_mean.reshape(
        1,
        3,
    )
)


# ============================================================
# SAVE NODAL PREDICTION
# ============================================================

prediction_dataframe = pd.DataFrame(
    {

        "NodeLabel":
            node_labels,

        "X":
            coordinates[
                :,
                0
            ],

        "Y":
            coordinates[
                :,
                1
            ],

        "Z":
            coordinates[
                :,
                2
            ],

        "U1":
            prediction[
                :,
                0
            ],

        "U2":
            prediction[
                :,
                1
            ],

        "U3":
            prediction[
                :,
                2
            ],
    }
)


prediction_dataframe.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("")
print(
    "Saved baseline DeepONet prediction:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# VALIDATE AGAINST SAME FULL FEM BASELINE
# ============================================================

if os.path.isfile(
    REFERENCE_FILE
):

    reference = pd.read_csv(
        REFERENCE_FILE
    )


    prediction_map = {}


    for _, row in prediction_dataframe.iterrows():

        key = (

            round(
                float(
                    row[
                        "X"
                    ]
                ),
                5,
            ),

            round(
                float(
                    row[
                        "Y"
                    ]
                ),
                5,
            ),

            round(
                float(
                    row[
                        "Z"
                    ]
                ),
                5,
            ),
        )


        prediction_map[
            key
        ] = np.array(
            [
                row[
                    "U1"
                ],
                row[
                    "U2"
                ],
                row[
                    "U3"
                ],
            ],
            dtype=np.float64,
        )


    predicted = []

    truth = []


    for _, row in reference.iterrows():

        key = (

            round(
                float(
                    row[
                        "X"
                    ]
                ),
                5,
            ),

            round(
                float(
                    row[
                        "Y"
                    ]
                ),
                5,
            ),

            round(
                float(
                    row[
                        "Z"
                    ]
                ),
                5,
            ),
        )


        if key not in prediction_map:

            raise RuntimeError(
                "Could not match reference node at {}".format(
                    key
                )
            )


        predicted.append(
            prediction_map[
                key
            ]
        )


        truth.append(
            np.array(
                [
                    row[
                        "U1"
                    ],
                    row[
                        "U2"
                    ],
                    row[
                        "U3"
                    ],
                ],
                dtype=np.float64,
            )
        )


    predicted = np.asarray(
        predicted
    )


    truth = np.asarray(
        truth
    )


    metric_rows = []


    for component_index, component_name in enumerate(
        [
            "U1",
            "U2",
            "U3",
        ]
    ):

        component_prediction = predicted[
            :,
            component_index
        ]


        component_truth = truth[
            :,
            component_index
        ]


        relative_l2 = (

            np.linalg.norm(
                component_prediction
                -
                component_truth
            )

            /

            (
                np.linalg.norm(
                    component_truth
                )
                +
                1.0e-14
            )
        )


        rmse = np.sqrt(
            np.mean(
                (
                    component_prediction
                    -
                    component_truth
                )
                ** 2
            )
        )


        mae = np.mean(
            np.abs(
                component_prediction
                -
                component_truth
            )
        )


        metric_rows.append(
            {

                "Component":
                    component_name,

                "Relative_L2":
                    relative_l2,

                "Relative_L2_percent":
                    100.0
                    *
                    relative_l2,

                "RMSE_mm":
                    rmse,

                "MAE_mm":
                    mae,
            }
        )


    metrics = pd.DataFrame(
        metric_rows
    )


    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )


    print("")
    print(
        "================================================"
    )

    print(
        "BASELINE FULL-DOMAIN DEEPONET vs FULL FEM"
    )

    print(
        "================================================"
    )


    print(
        metrics.to_string(
            index=False
        )
    )


    print("")
    print(
        "Saved:"
    )

    print(
        METRICS_FILE
    )


print("")
print(
    "Step 31A complete."
)
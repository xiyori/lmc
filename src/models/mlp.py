from torch import nn

from ..modules import Tracker, TrackedModel
from ..permutation_specs import PermutationSpec, permutation_spec_from_axes_to_perm


class MLPModel(TrackedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            Tracker(
                "Dense_0",
                nn.Linear(config.input_size, config.hidden_size),
                nn.ReLU(inplace=True),
                mode=config.matching_mode
            ),
            *[Tracker(
                f"Dense_{i}",
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                mode=config.matching_mode
            ) for i in range(1, config.num_layers - 1)],
            Tracker(
                f"Dense_{config.num_layers - 1}",
                nn.Linear(config.hidden_size, config.output_size),
                mode=config.matching_mode,
                enabled=False
            )
        )

        self.init()  # Trackers initialization

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.features(x)
        return x

    def copy(self):
        return MLPModel(self.config)


def mlp_permutation_spec(config) -> PermutationSpec:
    return permutation_spec_from_axes_to_perm({
        "Dense_0.weight": (0, None),
        **{f"Dense_{i}.weight": (i, i - 1)
           for i in range(1, config.num_layers - 1)},
        **{f"Dense_{i}.bias": (i,)
           for i in range(config.num_layers - 1)},
        f"Dense_{config.num_layers - 1}.weight": (None, config.num_layers - 2),
        f"Dense_{config.num_layers - 1}.bias": (None,)
    })

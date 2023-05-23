## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/
# based on https://gitlab.com/harvard-machine-learning/double-descent/blob/master/models/mcnn.py

from torch import nn
from torchvision import transforms

from ..modules import Indexer, IndexedModel
from ..permutation_specs import PermutationSpec, permutation_spec_from_axes_to_perm


def block(config, input, output, layer):
    # Layer i
    list = [
        Indexer(
            f"Conv_{layer}",
            nn.Conv2d(input, output, kernel_size=3,
                      stride=1, padding=1, bias=False),
            mode=config.matching_mode,
            track_activations=False
        ),
        Indexer(
            f"BatchNorm_{layer}",
            nn.BatchNorm2d(output),
            nn.ReLU(),
            mode=config.matching_mode
        ),
        nn.MaxPool2d(2)
    ]
    return list


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class ConvNetDepth(IndexedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.init_channels
        layer = 0
        module_list = block(config, 3, c, layer)
        layer += 1
        module_list = module_list[:-1]  # no max pooling at the end of first layer

        current_width = c
        last_zero = config.max_depth // 3 + 1 * (config.max_depth % 3 > 0) - 1
        for i in range(config.max_depth // 3 + 1 * (config.max_depth % 3 > 0)):
            if i != last_zero:
                module_list.extend(block(config, current_width, current_width, layer))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(config, current_width, 2 * current_width, layer))
                current_width = 2 * current_width
            layer += 1

        last_one = config.max_depth // 3 + 1 * (config.max_depth % 3 > 1) - 1
        for i in range(config.max_depth // 3 + 1 * (config.max_depth % 3 > 1)):
            if i != last_one:
                module_list.extend(block(config, current_width, current_width, layer))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(config, current_width, 2 * current_width, layer))
                current_width = 2 * current_width
            layer += 1

        last_two = config.max_depth // 3 + 1 * (config.max_depth % 3 > 2) - 1
        for i in range(config.max_depth // 3 + 1 * (config.max_depth % 3 > 2)):
            if i != last_two:
                module_list.extend(block(config, current_width, current_width, layer))
                module_list = module_list[:-1]  # no max pooling if we repeat layers
            else:
                module_list.extend(block(config, current_width, 2 * current_width, layer))
                current_width = 2 * current_width
            layer += 1

        pooling_increaser = 1
        if config.max_depth < 3:
            pooling_increaser = (3 - config.max_depth) * 2

        linear_layer = [
            nn.MaxPool2d(4 * pooling_increaser),
            Flatten(),
            Indexer(
                "Dense",
                nn.Linear(current_width, config.num_classes, bias=True),
                mode=config.matching_mode,
                track_activations=False
            )
        ]

        module_list.extend(linear_layer)

        self.module_list = nn.Sequential(*module_list)

        self.init()

    def forward(self, x):
        return self.module_list(x)

    def copy(self):
        return ConvNetDepth(self.config).to(self["Dense.weight"].device)


class ConvNet:
    base = ConvNetDepth
    args = []
    kwargs = {}
    transform_train = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def convnet_permutation_spec(config) -> PermutationSpec:
    num_layers = 1
    for i in range(3):
        num_layers += config.max_depth // 3 + 1 * (config.max_depth % 3 > i)
    return permutation_spec_from_axes_to_perm({
        "Conv_0.weight": (0, None, None, None),
        **{f"Conv_{i}.weight": (i, i - 1, None, None)
            for i in range(1, num_layers)},
        # **{f"Conv_{i}.bias": (i, )
        #     for i in range(num_layers)},
        **{f"BatchNorm_{i}.running_mean": (i, )
            for i in range(num_layers)},
        **{f"BatchNorm_{i}.running_var": (i, )
            for i in range(num_layers)},
        **{f"BatchNorm_{i}.weight": (i, )
            for i in range(num_layers)},
        **{f"BatchNorm_{i}.bias": (i, )
            for i in range(num_layers)},
        "Dense.weight": (None, num_layers - 1),
        "Dense.bias": (None, )
    })

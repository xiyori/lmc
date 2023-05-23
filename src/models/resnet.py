from torch import nn
from typing import Sequence

from ..modules import Indexer, IndexedModel
from ..permutation_specs import PermutationSpec, permutation_spec_from_axes_to_perm


class Block(nn.Module):
    def __init__(self, config, num_channels: int, stride: int, blockgroup: int, block: int):
        super().__init__()

        self.num_channels = num_channels
        self.stride = stride

        self.final_norm = Indexer(
            f"blockgroups_{blockgroup}/blocks_{block}/norm2",
            nn.BatchNorm2d(self.num_channels),
            nn.ReLU(),
            mode=config.matching_mode,
            track_activations=self.stride == 2
        )

        self.layers = nn.Sequential(
            Indexer(
                f"blockgroups_{blockgroup}/blocks_{block}/conv1",
                nn.Conv2d(self.num_channels // self.stride,
                          self.num_channels,
                          kernel_size=(3, 3),
                          stride=self.stride,
                          padding=1,
                          bias=False),
                mode=config.matching_mode,
                track_activations=False
            ),
            Indexer(
                f"blockgroups_{blockgroup}/blocks_{block}/norm1",
                nn.BatchNorm2d(self.num_channels),
                nn.ReLU(),
                mode=config.matching_mode
            ),
            Indexer(
                f"blockgroups_{blockgroup}/blocks_{block}/conv2",
                nn.Conv2d(self.num_channels,
                          self.num_channels,
                          kernel_size=(3, 3),
                          padding=1,
                          bias=False),
                mode=config.matching_mode,
                track_activations=False
            )
        )

        if self.stride != 1:
            assert self.stride == 2

            self.shortcut = nn.Sequential(
                Indexer(
                    f"blockgroups_{blockgroup}/blocks_{block}/shortcut/conv",
                    nn.Conv2d(
                        self.num_channels // 2,
                        self.num_channels,
                        kernel_size=(1, 1),
                        stride=self.stride,
                        bias=False),
                    mode=config.matching_mode,
                    track_activations=False
                ),
                Indexer(
                    f"blockgroups_{blockgroup}/blocks_{block}/shortcut/norm",
                    nn.BatchNorm2d(self.num_channels),
                    mode=config.matching_mode,
                    track_activations=False
                )
            )
        else:
            self.shortcut = lambda x: x

    def forward(self, x):
        x = self.layers(x) + self.shortcut(x)
        x = self.final_norm(x)
        return x


class BlockGroup(nn.Module):
    def __init__(self, config, num_channels: int, num_blocks: int, stride: int, blockgroup: int):
        super().__init__()

        assert num_blocks > 0
        self.blocks = nn.Sequential(
            Block(config, num_channels, stride, blockgroup, 0),
            *[Block(config, num_channels, 1, blockgroup, b) for b in range(1, num_blocks)]
        )

    def forward(self, x):
        return self.blocks(x)


class ResNet(IndexedModel):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.blocks_per_group: Sequence[int] = config.blocks_per_group
        self.num_classes: int = config.num_classes
        self.width_multiplier: int = config.width_multiplier
        wm = self.width_multiplier

        self.conv1 = Indexer(
            f"conv1",
            nn.Conv2d(3, 16 * wm,
                      kernel_size=(3, 3),
                      padding=1,
                      bias=False),
            mode=config.matching_mode,
            track_activations=False
        )
        self.norm1 = Indexer(
            f"norm1",
            nn.BatchNorm2d(16 * wm),
            nn.ReLU(),
            mode=config.matching_mode
        )

        channels_per_group = (16 * wm, 32 * wm, 64 * wm, 128 * wm)
        strides_per_group = (1, 2, 2, 2)
        self.blockgroups = nn.Sequential(*[
            BlockGroup(config, c, b, s, bg)
            for bg, (c, b, s) in enumerate(zip(
                channels_per_group, self.blocks_per_group, strides_per_group
            ))
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = Indexer(
            "dense",
            nn.Linear(128 * wm, self.num_classes),
            mode=config.matching_mode,
            track_activations=False
        )

        self.init()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.blockgroups(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.dense(x)
        return x

    def copy(self):
        return ResNet(self.config).to(self["dense.weight"].device)


def resnet18_permutation_spec() -> PermutationSpec:
    conv = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in, None, None)}
    norm = lambda name, p: {f"{name}.weight": (p, ), f"{name}.bias": (p, )}
    dense = lambda name, p_in, p_out: {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out, )}

    # This is for easy blocks that use a residual connection, without any change in the number of channels.
    easyblock = lambda name, p, p_inner: {
        **conv(f"{name}/conv1", p, p_inner),
        **norm(f"{name}/norm1", p_inner),
        **conv(f"{name}/conv2", p_inner, p),
        **norm(f"{name}/norm2", p)
    }

    # This is for blocks that use a residual connection, but change the number of channels via a Conv.
    shortcutblock = lambda name, p_in, p_out, p_inner: {
        **conv(f"{name}/conv1", p_in, p_inner),
        **norm(f"{name}/norm1", p_inner),
        **conv(f"{name}/conv2", p_inner, p_out),
        **norm(f"{name}/norm2", p_out),
        **conv(f"{name}/shortcut/conv", p_in, p_out),
        **norm(f"{name}/shortcut/norm", p_out)
    }

    return permutation_spec_from_axes_to_perm({
        **conv("conv1", None, 0),
        **norm("norm1", 0),
        #
        **easyblock("blockgroups_0/blocks_0", 0, 1),
        **easyblock("blockgroups_0/blocks_1", 0, 2),
        #
        **shortcutblock("blockgroups_1/blocks_0", 0, 3, 4),
        **easyblock("blockgroups_1/blocks_1", 3, 5),
        #
        **shortcutblock("blockgroups_2/blocks_0", 3, 6, 7),
        **easyblock("blockgroups_2/blocks_1", 6, 8),
        #
        **shortcutblock("blockgroups_3/blocks_0", 6, 9, 10),
        **easyblock("blockgroups_3/blocks_1", 9, 11),
        #
        **dense("dense", 9, None),
    })

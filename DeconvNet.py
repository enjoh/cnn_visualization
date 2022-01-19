import torch
import torch.nn as nn
from typing import List, Dict
from functools import partial


class DeconvNet(nn.Module):

    def __init__(self, from_conv=None) -> None:
        super().__init__()
        self.deconv_layers = nn.Sequential()
        self.conv_deconv_pairs: Dict[int, int] = {}
        self.conv_deconv_pairs_biases: Dict[int, int] = {}
        self.conv_indices: List = []
        self.deconv_indices: List = []
        self.pool_locs = {}
        self.maxpool_layers = {}
        self.feature_maps = {}
        if from_conv:
            self.from_conv(from_conv)

    def pool_hook(self, module, i, o, idx) -> None:
        """ hook to get maxpool locations """
        out, locs = self.maxpool_layers[idx](i[0])
        self.pool_locs[idx] = locs

    def conv_hook(self, module, i, o, idx) -> None:
        """ hook to get all feature maps """
        self.feature_maps[idx] = o.detach()

    def from_conv(self, conv_model) -> None:
        deconv_modules = nn.ModuleList()

        for i, m in enumerate(conv_model.features):
            if isinstance(m, nn.Conv2d):
                deconv_modules.insert(0, nn.Conv2d(
                    in_channels=m.out_channels,
                    out_channels=m.in_channels,
                    kernel_size=m.kernel_size,
                    padding=m.padding,
                    stride=m.stride))
                self.conv_indices.append(i)
                sz = len(conv_model.features)
                self.conv_deconv_pairs[i] = sz - i - 1
                self.deconv_indices.append(sz - i - 1)
                m.register_forward_hook(partial(self.conv_hook, idx=i))

            elif isinstance(m, nn.MaxPool2d):
                self.maxpool_layers[i] = nn.MaxPool2d(
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    return_indices=True,
                    dilation=m.dilation,
                    padding=m.padding,
                    ceil_mode=m.ceil_mode)
                m.register_forward_hook(partial(self.pool_hook, idx=i))

                deconv_modules.insert(0, nn.MaxUnpool2d(
                    kernel_size=m.kernel_size,
                    stride=m.stride))

            elif isinstance(m, nn.ReLU):
                deconv_modules.insert(0, nn.ReLU(
                    inplace=m.inplace))
            else:
                print("unknown layer")
                import sys
                sys.exit(0)

        self.conv_deconv_pairs_biases = dict(zip(self.conv_indices[0: -1:1], self.deconv_indices[1:-1:1]))
        del self.deconv_layers
        self.deconv_layers = nn.Sequential(*deconv_modules)
        self.initialize_weights(conv_model.features)

    def initialize_weights(self, src: nn.Sequential) -> None:
        for conv_layer, deconv_layer in self.conv_deconv_pairs.items():
            self.deconv_layers[deconv_layer].weight.data = torch.flip(torch.permute(src[conv_layer].weight.data, (1, 0, 2, 3)), (2, 3))
            if conv_layer in self.conv_deconv_pairs_biases.keys():
                if False:
                    # not really mentioned in paper, but without bias somehow looks "better"?
                    deconv_bias_index = self.conv_deconv_pairs_biases[conv_layer]
                    self.deconv_layers[deconv_bias_index].bias.data = torch.flip(src[conv_layer].bias.data, (0,))
                else:
                    deconv_bias_index = self.conv_deconv_pairs_biases[conv_layer]
                    self.deconv_layers[deconv_bias_index].bias.data = torch.zeros_like(self.deconv_layers[deconv_bias_index].bias.data)

    def forward(self, x: torch.Tensor, from_layer: int = -1) -> torch.Tensor:
        start_idx = self.conv_deconv_pairs[from_layer]
        ftrs = self.feature_maps[from_layer].clone()
        # sum up activations
        res = torch.sum(ftrs, dim=(2, 3))
        # get k strongest activations/feature maps and "kill" the rest
        cnt = 0
        _, idcs = res.topk(k=1, dim=1)
        for idx in range(ftrs.shape[1]):
            lst = idcs[0].tolist()
            if idx not in lst:
                ftrs[:, idx, :, :] = 0
            else:
                cnt += 1
        assert cnt == 1

        for i, layer in enumerate(self.deconv_layers[start_idx:]):
            if isinstance(layer, nn.MaxUnpool2d):
                locs = self.pool_locs[from_layer]
                ftrs = layer.forward(ftrs, locs)
            else:
                ftrs = layer.forward(ftrs)
            # keep track of current conv layer being looked at, to access correct maxpool locations
            from_layer -= 1
        return ftrs

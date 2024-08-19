from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, autograd, nn
from torch.nn.parameter import Parameter


# Straight through estimators
class STEThresh(autograd.Function):

    @staticmethod
    def forward(ctx, input: Tensor, thresh: float) -> Tensor:
        """
        Forward pass for the STEThresh function.

        Args:
            ctx: Context object to store information for backward
                computation.
            input (Tensor): Input tensor.
            thresh (float): Threshold value.

        Returns:
            Tensor: Output tensor after applying the threshold.
        """
        return (input > thresh).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        Backward pass for the STEThresh function.

        Args:
            ctx: Context object.
            grad_output (Tensor): Gradient of the output.

        Returns:
            Tuple[Tensor, None]: Gradient of the input and None for the threshold.
        """
        return grad_output, None


# Activation functions


class SPLASH(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, num_hinges: int = 5, init: str = "RELU"):
        """
        Initializes the SPLASH activation function.

        Args:
            num_hinges (int, optional): Number of hinges. Defaults to 5.
            init (str, optional): Initialization type ("RELU" or "LINEAR"). Defaults to "RELU".

        Raises:
            ValueError: If the initialization type is unknown.
        """
        super().__init__()
        assert num_hinges > 0, f"Number of hinges should be greater than zero, but is {num_hinges}"
        assert ((num_hinges + 1) % 2) == 0, f"Number of hinges should be odd, but is {num_hinges}"
        init = init.upper()

        self.num_hinges: int = num_hinges
        self.num_each_side: int = int((self.num_hinges + 1) / 2)

        self.hinges: List[float] = list(np.linspace(0, 2.5, self.num_each_side))

        self.output_bias: Parameter = Parameter(torch.zeros(1), requires_grad=True)

        if init == "RELU":
            self.coeffs_right: Parameter = Parameter(torch.cat(
                (torch.ones(1), torch.zeros(self.num_each_side - 1))),
                                                     requires_grad=True)
            self.coeffs_left: Parameter = Parameter(torch.zeros(self.num_each_side),
                                                    requires_grad=True)
        elif init == "LINEAR":
            self.coeffs_right: Parameter = Parameter(torch.cat(
                (torch.ones(1), torch.zeros(self.num_each_side - 1))),
                                                     requires_grad=True)
            self.coeffs_left: Parameter = Parameter(
                torch.cat((-torch.ones(1), torch.zeros(self.num_each_side - 1))),
                requires_grad=True,
            )
        else:
            raise ValueError(f"Unknown init {init}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the SPLASH activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the SPLASH activation.
        """
        output: Tensor = torch.zeros_like(x)

        # output for x > 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_right[idx] * torch.clamp(x - self.hinges[idx], min=0)

        # output for x < 0
        for idx in range(self.num_each_side):
            output = output + self.coeffs_left[idx] * torch.clamp(-x - self.hinges[idx], min=0)

        output = output + self.output_bias

        return output


class LinearAct(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self):
        """
        Initializes the linear activation function.
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the linear activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor (same as input).
        """
        return x


def get_act_fn(act: str) -> nn.Module:
    """
    Returns the activation function based on the given string.

    Args:
        act (str): Activation function name.

    Returns:
        nn.Module: Corresponding activation function.

    Raises:
        ValueError: If the activation type is undefined.
    """
    act = act.upper()
    if act == "RELU":
        act_fn = nn.ReLU()
    elif act == "ELU":
        act_fn = nn.ELU()
    elif act == "SIGMOID":
        act_fn = nn.Sigmoid()
    elif act == "TANH":
        act_fn = nn.Tanh()
    elif act == "SPLASH":
        act_fn = SPLASH()
    elif act == "LINEAR":
        act_fn = LinearAct()
    elif act == "LRELU":
        act_fn = nn.LeakyReLU()
    else:
        raise ValueError(f"Un-defined activation type {act}")

    return act_fn


class FullyConnectedModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self,
                 input_dim: int,
                 layer_dims: List[int],
                 layer_batch_norms: List[bool],
                 layer_acts: List[str],
                 weight_norms: Optional[List[bool]] = None,
                 layer_norms: Optional[List[bool]] = None,
                 dropouts: Optional[List[float]] = None,
                 use_bias_with_norm: bool = True):
        """
        Initializes the FullyConnectedModel.

        Args:
            input_dim (int): Input dimension.
            layer_dims (List[int]): List of layer dimensions.
            layer_batch_norms (List[bool]): List of batch normalization flags.
            layer_acts (List[str]): List of activation functions.
            weight_norms (Optional[List[bool]], optional): List of weight normalization flags.
                Defaults to None.
            layer_norms (Optional[List[bool]], optional): List of layer normalization flags.
                Defaults to None.
            dropouts (Optional[List[float]], optional): List of dropout rates. Defaults to None.
            use_bias_with_norm (bool, optional): Whether to use bias with normalization. Defaults
                to True.
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        if weight_norms is None:
            weight_norms = [False] * len(layer_dims)

        if layer_norms is None:
            layer_norms = [False] * len(layer_dims)

        if dropouts is None:
            dropouts = [0.0] * len(layer_dims)

        # layers
        for layer_dim, batch_norm, act, weight_norm, layer_norm, dropout in zip(
                layer_dims, layer_batch_norms, layer_acts, weight_norms, layer_norms, dropouts):
            module_list = nn.ModuleList()

            # linear
            use_bias = use_bias_with_norm or (not batch_norm and not layer_norm)
            linear_layer = nn.Linear(input_dim, layer_dim, bias=use_bias)
            if weight_norm:
                linear_layer = nn.utils.weight_norm(linear_layer)

            module_list.append(linear_layer)

            # layer normalization
            if layer_norm:
                module_list.append(nn.LayerNorm(layer_dim))

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm1d(layer_dim))

            # activation
            module_list.append(get_act_fn(act))

            # dropout
            if dropout > 0.0:
                module_list.append(nn.Dropout(dropout))

            self.layers.append(module_list)

            input_dim = layer_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the FullyConnectedModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the fully connected layers.
        """
        x = x.float()
        module_list: nn.ModuleList

        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


class ResnetModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self,
                 resnet_dim: int,
                 num_resnet_blocks: int,
                 out_dim: int,
                 batch_norm: bool,
                 act: str,
                 use_bias_with_norm: bool = True):
        """
        Initializes the ResnetModel.

        Args:
            resnet_dim (int): Dimension of the ResNet blocks.
            num_resnet_blocks (int): Number of ResNet blocks.
            out_dim (int): Output dimension.
            batch_norm (bool): Whether to use batch normalization.
            act (str): Activation function.
            use_bias_with_norm (bool, optional): Whether to use bias with normalization. Defaults
                to True.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        self.block_act_fns = nn.ModuleList()

        # resnet blocks
        for _ in range(num_resnet_blocks):
            block_net = FullyConnectedModel(resnet_dim, [resnet_dim] * 2, [batch_norm] * 2,
                                            [act, "LINEAR"],
                                            use_bias_with_norm=use_bias_with_norm)
            module_list: nn.ModuleList = nn.ModuleList([block_net])

            self.blocks.append(module_list)
            self.block_act_fns.append(get_act_fn(act))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the ResnetModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResNet blocks and the final linear
                layer.
        """
        # resnet blocks
        module_list: nn.ModuleList
        for module_list, act_fn in zip(self.blocks, self.block_act_fns):
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        # output
        x = self.fc_out(x)
        return x


class Conv2dModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self,
                 chan_in: int,
                 channel_sizes: List[int],
                 kernel_sizes: List[Union[int, Tuple[int, int]]],
                 paddings: List[Union[int, Tuple[int, int], str]],
                 layer_batch_norms: List[bool],
                 layer_acts: List[str],
                 strides: Optional[List[Union[int, Tuple[int, int]]]] = None,
                 transpose: bool = False,
                 weight_norms: Optional[List[bool]] = None,
                 poolings: List[Optional[str]] = None,
                 dropouts: Optional[List[float]] = None,
                 padding_modes: Optional[List[str]] = None,
                 padding_values: Optional[List[Union[int, float]]] = None,
                 group_norms: Optional[List[int]] = None,
                 use_bias_with_norm: bool = True):
        """
        Initializes the Conv2dModel.

        Args:
            chan_in (int): Number of input channels.
            channel_sizes (List[int]): List of output channel sizes.
            kernel_sizes (List[Union[int, Tuple[int, int]]]): List of kernel sizes.
            paddings (List[Union[int, Tuple[int, int], str]]): List of paddings.
            layer_batch_norms (List[bool]): List of batch normalization flags.
            layer_acts (List[str]): List of activation functions.
            strides (Optional[List[Union[int, Tuple[int, int]]]], optional): List of strides.
                Defaults to None.
            transpose (bool, optional): Whether to use transposed convolution. Defaults to False.
            weight_norms (Optional[List[bool]], optional): List of weight normalization flags.
                Defaults to None.
            poolings (List[Optional[str]], optional): List of pooling types. Defaults to None.
            dropouts (Optional[List[float]], optional): List of dropout rates. Defaults to None.
            padding_modes (Optional[List[str]], optional): List of padding modes. 'none', 'zeros',
                'reflect', 'replicate', 'circular' or 'constant'. Defaults to 'zeros'. 'none' will
                use 'zeros' as well, but if padding = 0, it doesn't matter.
            padding_values (Optional[List[Union[int, float]]], optional): List of padding values.
                if padding mode = 'constant' padding will be filled with 'value' if specified,
                otherwise 'zero'. Defaults to None.
            group_norms (Optional[List[int]], optional): List of group normalization values.
                Defaults to None.
            use_bias_with_norm (bool, optional): Whether to use bias with normalization. Defaults
                to True.
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        if strides is None:
            strides = [1] * len(channel_sizes)

        if weight_norms is None:
            weight_norms = [False] * len(channel_sizes)

        if dropouts is None:
            dropouts = [0.0] * len(channel_sizes)

        if group_norms is None:
            group_norms = [0] * len(channel_sizes)

        if padding_modes is None:
            padding_modes = ["zeros"] * len(channel_sizes)

        if padding_values is None:
            padding_values = [0] * len(channel_sizes)

        if poolings is None:
            poolings = [None] * len(channel_sizes)

        # layers
        for (chan_out, kernel_size, padding, batch_norm, act, stride, weight_norm, dropout,
             padding_mode, padding_value, group_norm,
             pooling) in zip(channel_sizes, kernel_sizes, paddings, layer_batch_norms, layer_acts,
                             strides, weight_norms, dropouts, padding_modes, padding_values,
                             group_norms, poolings):

            module_list = nn.ModuleList()

            if padding_mode == "none":
                padding_mode = "zeros"

            elif padding_mode == "constant":
                assert isinstance(padding, int), "'padding' must be an integer in 'constant' mode."
                padding_layer = nn.ConstantPad2d(padding, padding_value)
                module_list.append(padding_layer)
                padding_mode = "zeros"
                padding = 0

            # Conv
            use_bias = use_bias_with_norm or (not batch_norm and not group_norm)
            if transpose:
                conv_layer = nn.ConvTranspose2d(chan_in,
                                                chan_out,
                                                kernel_size,
                                                padding=padding,
                                                stride=stride,
                                                padding_mode=padding_mode,
                                                bias=use_bias)
            else:
                conv_layer = nn.Conv2d(chan_in,
                                       chan_out,
                                       kernel_size,
                                       padding=padding,
                                       stride=stride,
                                       padding_mode=padding_mode,
                                       bias=use_bias)

            if weight_norm:
                conv_layer = nn.utils.weight_norm(conv_layer)

            module_list.append(conv_layer)

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm2d(chan_out))

            # if group_norm = 0, no group normalization
            # if group_notm = 1, same as layer normalization
            # if group_norm = number of channels, same as instance normalization
            # otherwise, is a group normalization
            elif group_norm > 0:
                # Check if number of channels is divisible by number of groups
                assert (chan_out % group_norm == 0
                        ), f"chan_out ({chan_out}) must be divisible by group_norm ({group_norm})"
                module_list.append(nn.GroupNorm(group_norm, chan_out))

            # activation
            module_list.append(get_act_fn(act))

            # dropout
            if dropout > 0.0:
                module_list.append(nn.Dropout(dropout))

            if not transpose and pooling == "avg":
                module_list.append(nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
            elif transpose and pooling == "max":
                module_list.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))

            self.layers.append(module_list)

            chan_in = chan_out

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Conv2dModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the convolutional layers.
        """
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


class ResnetConv2dModel(nn.Module):

    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self,
                 in_channels: int,
                 resnet_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: Union[int, str],
                 num_resnet_blocks: int,
                 batch_norm: bool,
                 act: str,
                 group_norm: Optional[int] = 0,
                 use_bias_with_norm: bool = True) -> None:
        """
        Initializes the ResnetConv2dModel.

        Args:
            in_channels (int): Number of input channels.
            resnet_channels (int): Number of ResNet channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            padding (Union[int, str]): Padding value or type.
            num_resnet_blocks (int): Number of residual blocks.
            batch_norm (bool): Whether to use batch normalization.
            act (str): Activation function.
            group_norm (Optional[int], optional): Group normalization value. Defaults to 0.
            use_bias_with_norm (bool, optional): Whether to use bias with normalization. Defaults
                to True.
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.block_act_fns: nn.ModuleList = nn.ModuleList()
        self.downsample: Optional[Conv2dModel] = None
        self.first_layer: Optional[Conv2dModel] = None

        self.needs_downsampling: bool = resnet_channels != out_channels
        self.needs_shape_match: bool = in_channels != resnet_channels

        if kernel_size == 2 and padding == "same":
            paddings: List[Union[int, str]] = [1, 0]
        else:
            paddings = [padding] * 2

        # match the channels shape
        if self.needs_shape_match:
            self.first_layer = Conv2dModel(
                in_channels,
                [resnet_channels],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[group_norm],
                use_bias_with_norm=use_bias_with_norm,
            )

        # resnet blocks
        for _ in range(num_resnet_blocks):
            block_net = Conv2dModel(
                resnet_channels,
                [resnet_channels] * 2,
                [kernel_size] * 2,
                paddings,
                [batch_norm] * 2,
                [act, "LINEAR"],
                group_norms=[group_norm] * 2,
                use_bias_with_norm=use_bias_with_norm,
            )

            module_list: nn.ModuleList = nn.ModuleList([block_net])

            self.blocks.append(module_list)
            self.block_act_fns.append(get_act_fn(act))

        if self.needs_downsampling:
            self.downsample = Conv2dModel(
                resnet_channels,
                [out_channels],
                [1],
                [0],
                [False],
                ["RELU"],
                group_norms=[group_norm],
                use_bias_with_norm=use_bias_with_norm,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the ResnetConv2dModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the residual blocks and any necessary
                downsampling.
        """
        if self.needs_shape_match:
            # match the channels shape
            x = self.first_layer(x)

        # resnet blocks
        module_list: nn.ModuleList
        for module_list, act_fn in zip(self.blocks, self.block_act_fns):
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        if self.needs_downsampling:
            x = self.downsample(x)

        return x

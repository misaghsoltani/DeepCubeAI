from __future__ import annotations

from typing import cast

import numpy as np
import torch
from torch import Tensor, autograd, nn
from torch.autograd.function import FunctionCtx
from torch.nn.parameter import Parameter


# Straight through estimators
class STEThresh(autograd.Function):
    """Straight Through Estimator for thresholding."""

    @staticmethod
    def forward(ctx: FunctionCtx, input: Tensor, thresh: float) -> Tensor:
        """Forward pass for the STEThresh function.

        Args:
            ctx: Context object to store information for backward computation.
            input (Tensor): Input tensor.
            thresh (float): Threshold value.

        Returns:
            Tensor: Output tensor after applying the threshold.
        """
        return (input > thresh).float()

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Backward pass for the STEThresh function.

        Args:
            ctx: Context object.
            grad_output (Tensor): Gradient of the output.

        Returns:
            tuple[Tensor, None]: Gradient of the input and None for the threshold.
        """
        return grad_output, None


# Activation functions


class SPLASH(nn.Module):
    """SPLASH activation function."""

    def __init__(self, num_hinges: int = 5, init: str = "RELU") -> None:
        """Initializes the SPLASH activation function.

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

        self.hinges: list[float] = list(np.linspace(0, 2.5, self.num_each_side))

        self.output_bias: Parameter = Parameter(torch.zeros(1), requires_grad=True)

        if init == "RELU":
            coeffs_right_values = torch.cat((torch.ones(1), torch.zeros(self.num_each_side - 1)))
            coeffs_left_values = torch.zeros(self.num_each_side)
        elif init == "LINEAR":
            coeffs_right_values = torch.cat((torch.ones(1), torch.zeros(self.num_each_side - 1)))
            coeffs_left_values = torch.cat((-torch.ones(1), torch.zeros(self.num_each_side - 1)))
        else:
            raise ValueError(f"Unknown init {init}")

        self.coeffs_right: Parameter = Parameter(coeffs_right_values, requires_grad=True)
        self.coeffs_left: Parameter = Parameter(coeffs_left_values, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the SPLASH activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the SPLASH activation.
        """
        output: Tensor = torch.zeros_like(x)

        # output for x > 0
        for idx in range(self.num_each_side):
            output += self.coeffs_right[idx] * torch.clamp(x - self.hinges[idx], min=0)

        # output for x < 0
        for idx in range(self.num_each_side):
            output += self.coeffs_left[idx] * torch.clamp(-x - self.hinges[idx], min=0)

        output += self.output_bias

        return output


def get_act_fn(act: str) -> nn.Module:
    """Gets the activation function based on the input string.

    Args:
        act (str): Activation function name.

    Returns:
        nn.Module: The activation function module.
    """
    act = act.upper()
    if act == "RELU":
        act_fn: nn.Module = nn.ReLU()
    elif act == "ELU":
        act_fn = nn.ELU()
    elif act == "SIGMOID":
        act_fn = nn.Sigmoid()
    elif act == "TANH":
        act_fn = nn.Tanh()
    elif act == "SPLASH":
        act_fn = SPLASH()
    elif act == "LINEAR":
        act_fn = nn.Identity()
    elif act == "LRELU":
        act_fn = nn.LeakyReLU()
    else:
        raise ValueError(f"Un-defined activation type {act}")

    return act_fn


class FullyConnectedModel(nn.Module):
    """A fully connected neural network model."""

    def __init__(
        self,
        input_dim: int,
        layer_dims: list[int],
        layer_batch_norms: list[bool],
        layer_acts: list[str],
        weight_norms: list[bool] | None = None,
        layer_norms: list[bool] | None = None,
        dropouts: list[float] | None = None,
        use_bias_with_norm: bool = True,
    ) -> None:
        """Initializes the FullyConnectedModel.

        Args:
            input_dim (int): Input dimension.
            layer_dims (list[int]): List of layer dimensions.
            layer_batch_norms (list[bool]): List of batch normalization flags.
            layer_acts (list[str]): List of activation functions.
            weight_norms (Optional[list[bool]], optional): List of weight normalization flags.
                Defaults to None.
            layer_norms (Optional[list[bool]], optional): List of layer normalization flags.
                Defaults to None.
            dropouts (Optional[list[float]], optional): List of dropout rates. Defaults to None.
            use_bias_with_norm (bool, optional): Whether to use bias if the is a normalization immidiately
                after the
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
            layer_dims, layer_batch_norms, layer_acts, weight_norms, layer_norms, dropouts, strict=False
        ):
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
        """Forward pass for the FullyConnectedModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the fully connected layers.
        """
        for layer in self.layers:
            module_list = cast(nn.ModuleList, layer)
            for module in module_list:
                x = module(x)

        return x


class ResnetModel(nn.Module):
    """Residual Network model for neural network architecture.

    A ResNet implementation with customizable residual blocks, dimensionality,
    and activation functions.
    """

    def __init__(
        self,
        resnet_dim: int,
        num_resnet_blocks: int,
        out_dim: int,
        batch_norm: bool,
        act: str,
        use_bias_with_norm: bool = True,
    ) -> None:
        """Initializes the ResnetModel.

        Args:
            resnet_dim (int): Dimension of the ResNet blocks.
            num_resnet_blocks (int): Number of ResNet blocks.
            out_dim (int): Output dimension.
            batch_norm (bool): Whether to use batch normalization.
            act (str): Activation function.
            use_bias_with_norm (bool, optional): Whether to use bias if the is a normalization immidiately
                after the
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.block_act_fns: nn.ModuleList = nn.ModuleList()

        # resnet blocks
        for _ in range(num_resnet_blocks):
            block_net = FullyConnectedModel(
                resnet_dim, [resnet_dim] * 2, [batch_norm] * 2, [act, "LINEAR"], use_bias_with_norm=use_bias_with_norm
            )
            module_list: nn.ModuleList = nn.ModuleList([block_net])

            self.blocks.append(module_list)
            self.block_act_fns.append(get_act_fn(act))

        # output
        self.fc_out: nn.Module = nn.Linear(resnet_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the ResnetModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResNet blocks and the final linear
                layer.
        """
        # resnet blocks
        for block, act_fn in zip(self.blocks, self.block_act_fns, strict=False):
            module_list = cast(nn.ModuleList, block)
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        # output
        x = self.fc_out(x)
        return x


class Conv2dModel(nn.Module):
    """2D Convolutional neural network model.

    A flexible CNN implementation supporting various layer configurations,
    batch normalization, activation functions, and transpose convolutions.
    """

    def __init__(
        self,
        chan_in: int,
        channel_sizes: list[int],
        kernel_sizes: list[int | tuple[int, int]],
        paddings: list[int | tuple[int, int] | str],
        layer_batch_norms: list[bool],
        layer_acts: list[str],
        strides: list[int | tuple[int, int]] | None = None,
        transpose: bool = False,
        weight_norms: list[bool] | None = None,
        poolings: list[str | None] | None = None,
        dropouts: list[float] | None = None,
        padding_modes: list[str] | None = None,
        padding_values: list[int | float] | None = None,
        group_norms: list[int] | None = None,
        use_bias_with_norm: bool = True,
    ) -> None:
        """Initializes the Conv2dModel.

        Args:
            chan_in (int): Number of input channels.
            channel_sizes (list[int]): List of output channel sizes.
            kernel_sizes (list[Union[int, tuple[int, int]]]): List of kernel sizes.
            paddings (list[Union[int, tuple[int, int], str]]): List of paddings.
            layer_batch_norms (list[bool]): List of batch normalization flags.
            layer_acts (list[str]): List of activation functions.
            strides (Optional[list[Union[int, tuple[int, int]]]], optional): List of strides. Defaults to None.
            transpose (bool, optional): Whether to use transposed convolution. Defaults to False.
            weight_norms (Optional[list[bool]], optional): List of weight normalization flags. Defaults to None.
            poolings (list[Optional[str]], optional): List of pooling types. Defaults to None.
            dropouts (Optional[list[float]], optional): List of dropout rates. Defaults to None.
            padding_modes (Optional[list[str]], optional): List of padding modes. 'none', 'zeros',
                'reflect', 'replicate', 'circular' or 'constant'. Defaults to 'zeros'. 'none' will
                use 'zeros' as well, but if padding = 0, it doesn't matter.
            padding_values (Optional[list[Union[int, float]]], optional): List of padding values.
                if padding mode = 'constant' padding will be filled with 'value' if specified,
                otherwise 'zero'. Defaults to None.
            group_norms (Optional[list[int]], optional): List of number of groups for group normalization.
                Defaults to None.
            use_bias_with_norm (bool, optional): Whether to use bias if there is a normalization
                (such as BatchNorm or GroupNorm) used in this layer.
        """
        super().__init__()
        self.layers: nn.ModuleList = nn.ModuleList()
        if strides is None:
            strides = cast(list[int | tuple[int, int]], [1] * len(channel_sizes))

        if weight_norms is None:
            weight_norms = [False] * len(channel_sizes)

        if dropouts is None:
            dropouts = [0.0] * len(channel_sizes)

        if group_norms is None:
            group_norms = [0] * len(channel_sizes)

        if padding_modes is None:
            padding_modes = ["zeros"] * len(channel_sizes)

        if padding_values is None:
            padding_values = cast(list[int | float], [0] * len(channel_sizes))

        if poolings is None:
            poolings = cast(list[str | None], [None] * len(channel_sizes))

        # Ensure all lists are not None after assignment
        assert strides is not None
        assert padding_values is not None
        assert poolings is not None

        # layers
        for (
            chan_out,
            kernel_size,
            padding,
            batch_norm,
            act,
            stride,
            weight_norm,
            dropout,
            padding_mode,
            padding_value,
            group_norm,
            pooling,
        ) in zip(
            channel_sizes,
            kernel_sizes,
            paddings,
            layer_batch_norms,
            layer_acts,
            strides,
            weight_norms,
            dropouts,
            padding_modes,
            padding_values,
            group_norms,
            poolings,
            strict=False,
        ):
            module_list = nn.ModuleList()

            current_padding_mode = padding_mode
            current_padding = padding

            if current_padding_mode == "none":
                current_padding_mode = "zeros"

            elif current_padding_mode == "constant":
                assert isinstance(current_padding, int), "'padding' must be an integer in 'constant' mode."
                padding_layer = nn.ConstantPad2d(current_padding, padding_value)
                module_list.append(padding_layer)
                current_padding_mode = "zeros"
                current_padding = 0

            # Conv
            conv_layer: nn.Conv2d | nn.ConvTranspose2d
            use_bias = use_bias_with_norm or (not batch_norm and not group_norm)
            if transpose:
                conv_layer = nn.ConvTranspose2d(
                    chan_in,
                    chan_out,
                    kernel_size,
                    padding=current_padding,
                    stride=stride,
                    padding_mode=current_padding_mode,
                    bias=use_bias,
                )
            else:
                conv_layer = nn.Conv2d(
                    chan_in,
                    chan_out,
                    kernel_size,
                    padding=current_padding,
                    stride=stride,
                    padding_mode=current_padding_mode,
                    bias=use_bias,
                )

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
                assert chan_out % group_norm == 0, (
                    f"chan_out ({chan_out}) must be divisible by group_norm ({group_norm})"
                )
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
        """Forward pass for the Conv2dModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the convolutional layers.
        """
        x = x.float()

        for layer in self.layers:
            module_list = cast(nn.ModuleList, layer)
            for module in module_list:
                x = module(x)

        return x


class ResnetConv2dModel(nn.Module):
    """ResNet-based 2D Convolutional model.

    Combines ResNet architecture with 2D convolutions for handling
    image-like data with residual connections.
    """

    def __init__(
        self,
        in_channels: int,
        resnet_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | str,
        num_resnet_blocks: int,
        batch_norm: bool,
        act: str,
        group_norm: int | None = 0,
        use_bias_with_norm: bool = True,
    ) -> None:
        """Initializes the ResnetConv2dModel.

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
            use_bias_with_norm (bool, optional): Whether to use bias if the is a normalization immidiately
                after the
        """
        super().__init__()
        self.blocks: nn.ModuleList = nn.ModuleList()
        self.block_act_fns: nn.ModuleList = nn.ModuleList()
        self.downsample: Conv2dModel | None = None
        self.first_layer: Conv2dModel | None = None

        self.needs_downsampling: bool = resnet_channels != out_channels
        self.needs_shape_match: bool = in_channels != resnet_channels

        if kernel_size == 2 and padding == "same":
            paddings: list[int | str] = [1, 0]
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
                group_norms=[group_norm] if group_norm is not None else None,
                use_bias_with_norm=use_bias_with_norm,
            )

        # resnet blocks
        for _ in range(num_resnet_blocks):
            block_net = Conv2dModel(
                resnet_channels,
                [resnet_channels] * 2,
                [kernel_size] * 2,
                cast(list[int | tuple[int, int] | str], paddings),
                [batch_norm] * 2,
                [act, "LINEAR"],
                group_norms=[group_norm, group_norm] if group_norm is not None else None,
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
                group_norms=[group_norm] if group_norm is not None else None,
                use_bias_with_norm=use_bias_with_norm,
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the ResnetConv2dModel.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the residual blocks and any necessary
                downsampling.
        """
        if self.needs_shape_match:
            # match the channels shape
            assert self.first_layer is not None
            x = self.first_layer(x)

        # resnet blocks
        for block, act_fn in zip(self.blocks, self.block_act_fns, strict=False):
            module_list = cast(nn.ModuleList, block)
            res_inp = x
            for module in module_list:
                x = module(x)

            x = act_fn(x + res_inp)

        if self.needs_downsampling:
            assert self.downsample is not None
            x = self.downsample(x)

        return x

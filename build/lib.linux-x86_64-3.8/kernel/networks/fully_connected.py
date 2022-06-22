from torch import nn, Tensor
from typing import Iterable
import torch


class FCNetwork(nn.Module):
    """Fully connected PyTorch neural network class

    :attr input_size (int): dimensionality of input tensors
    :attr out_size (int): dimensionality of output tensors
    :attr layers (torch.nn.Module): neural network as sequential network of multiple layers
    """

    def __init__(self, dims: Iterable[int], output_activation: nn.Module = None, **kwargs):
        """Creates a network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        """
        super().__init__()
        self.input_size = dims[0]
        self.out_size = dims[-1]
        self.layers = self.make_seq(dims, output_activation, kwargs['dropout_prob'])
        self.config = kwargs

    @staticmethod
    def make_seq(dims: Iterable[int], output_activation: nn.Module, dropout_prob) -> nn.Module:
        """Creates a sequential network using ReLUs between layers and no activation at the end

        :param dims (Iterable[int]): tuple in the form of (IN_SIZE, HIDDEN_SIZE, HIDDEN_SIZE2,
            ..., OUT_SIZE) for dimensionalities of layers
        :param output_activation (nn.Module): PyTorch activation function to use after last layer
        :return (nn.Module): return created sequential layers


        - input norm ? - not neccessary with batch norm as first layer
        - batch norm
        - dropout


        """
        mods = []

        for i in range(len(dims) - 2): # TODO replace this with more intuative generation mechanism
            mods.append(nn.BatchNorm1d(dims[i]))
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            mods.append(nn.Tanh())
            mods.append(nn.Dropout(p=dropout_prob)) # TODO ------------------------- ensure this doesnt drop out during evaluations

        # mods.append(nn.Linear(dims[-3], dims[-2]))
        # mods.append(nn.ReLU())

        # for i in range(int(len(dims)/2)+1, len(dims) - 2 ): # TODO replace this with more intuative generation mechanism
        #     mods.append(nn.BatchNorm1d(dims[i]))
        #     mods.append(nn.Linear(dims[i], dims[i + 1]))
        #     mods.append(nn.Tanh())
        #     mods.append(nn.Dropout(p=dropout_prob)) # TODO ------------------------- ensure this doesnt drop out during evaluations

        mods.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation:
            mods.append(output_activation())

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        net = nn.Sequential(*mods)
        net.apply(init_weights)
        return net

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass through the network

        :param x (torch.Tensor): input tensor to feed into the network
        :return (torch.Tensor): output computed by the network
        """
        # Feedforward
        return self.layers(x)

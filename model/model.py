""" MLP in pytorch
"""

import torch as T
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Type, Optional

# -- Contrastive MLP 
class Residual(nn.Module):
    '''
     Resiual layer for FF netoworks, resizes original input using linear transformation
    '''
    def __init__(self, 
                 layer: Type[nn.Module],
                 in_dim: int, 
                 out_dim: Optional[int] = None,
                 layer_scale: Optional[float]= None,
                 ) -> None:
        
        super().__init__()
        self.layer = layer
        out_dim = out_dim or in_dim  # use input size if output size not provided
        self.layer_scale = nn.Parameter(T.ones(out_dim,) * layer_scale) if layer_scale is not None else 1.0
        
        if in_dim == out_dim:
            self.resize = nn.Identity()  # dont resize residual if input and output are same size
        else:
            self.resize = nn.Linear(in_dim, out_dim)  # otherwise resize with linear layer
    
    def forward(self, x):
        return (self.layer(x) * self.layer_scale) + self.resize(x)  # residual connection

class Norm(nn.Module):
    ''' Tabular layernorm implementation
    '''
    
    def forward(self, x):
        return F.normalise(x, p = 2., dim = -1)
    
class DenseBlock(nn.Module):
    '''
     Dense blocks for feedforward network
    '''
    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        activation: Callable[[], nn.Module] = nn.ReLU,
        residual: bool = False,
        layer_scale: Optional[float] = None,
        layernorm: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim or in_dim  # use input dimenstions if output size not provided
        
        norm_layer = layernorm or nn.Identity
        
        # linear layer with activation 
        self.block = nn.Sequential(nn.Linear(in_dim, out_dim, bias = bias), activation(), norm_layer(out_dim), dropout_layer(dropout))
        
        if residual:  # add residual connection
            self.block = Residual(self.block, in_dim, out_dim, layer_scale=layer_scale)
        
    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.block(x)
    
def make_mlp(
        d_in: int,
        neurons: List[int],
        activation: Callable[[],nn.Module] = nn.ReLU,
        dropout: float = 0.,
        residual: List[bool] = [],
        norm_layer: Callable[[int],nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        d_out: Optional[int] = None,
        final_layer_activation: Optional[Callable[[], nn.Module]] = nn.ReLU,
        layer_scale: Optional[float] = None,
        block_norm: bool = False,
        bias: bool = True,
    ) -> nn.Sequential:
    
    if not isinstance(neurons, list):
        neurons = [neurons]

    neurons = [d_in] + neurons  # prepend input size to list of neurons
    if d_out is not None:
        neurons = neurons + [d_out]
    layers = []  # init layers
    
    if isinstance(residual, bool):
        residual = [residual for _ in neurons]
    
    if len(residual) == 0:
        residual = [False for _ in neurons]
        
    elif len(residual) != len(neurons):
        raise ValueError('Length of residual argument must match length of neurons argument got, residual len: {len(residula)}, neuron len: {len(neurons)}')
    
    # iterate over layers
    for i in range(len(neurons) - 1):
        # add layer with actiation function and dropout
        if i != len(neurons) - 2:
            # add dense block
            layers += [DenseBlock(neurons[i], neurons[i + 1], activation = activation, residual = residual[i], dropout = dropout, dropout_layer = dropout_layer, layer_scale=layer_scale, layernorm = norm_layer if block_norm else None, bias = bias)]
        
        else:  # final layer has no activation or dropout        
            layers += [DenseBlock(neurons[i], neurons[i + 1], activation = nn.Identity if final_layer_activation is None else final_layer_activation, residual = residual[i], dropout_layer=nn.Identity, layer_scale=layer_scale,layernorm = norm_layer if block_norm else None, bias = bias)]
    
    # use layernorm on output if required
    if not block_norm:
        layers += [norm_layer(neurons[-1])]

    return nn.Sequential(*layers)  # create sequential network from layers

class ContrastiveMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        neurons: List[int],
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
        residual: List[bool] = [],
        norm_layer: Callable[[int],nn.Module] = nn.Identity,
        dropout_layer: Callable[[int], nn.Module] = nn.Dropout,
        n_classes: Optional[int] = None, 
        d_out: Optional[int] = None,
        final_layer_activation: Optional[Callable[[], nn.Module]] = nn.ReLU,
        project_to_sphere: bool = False,
        layer_scale: float = None,
        block_norm: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        self.proj_to_sphere = project_to_sphere
        
        # make mlp
        norm_layer = Identity if norm_layer is None else norm_layer
        neurons = neurons if isinstance(neurons,list) else [neurons]
        self.mlp = make_mlp(
            d_in = d_in,
            neurons = neurons,
            activation = activation,
            dropout = dropout,
            residual=residual,
            norm_layer=norm_layer,
            dropout_layer=dropout_layer,
            final_layer_activation=final_layer_activation,
            layer_scale= layer_scale,
            block_norm = block_norm,
            bias = bias,
        )
        
        # linear projection heads
        self.proj = nn.Identity() if d_out is None else nn.Linear(neurons[-1], d_out)
        self.probe = nn.Identity() if n_classes is None else nn.Linear(neurons[-1], n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.mlp(x)
    
    def forward_cls(self, x: Tensor) -> Tensor:
        return self.probe(x)

    def forward_finetune(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_cls(x)
        return x

    def forward_proj(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        
        if self.proj_to_sphere:
            x = F.normalize(x, dim = -1)
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_proj(self.forward_features(x))
    
"""
Implementation of a range of Graph Recurrent Networks.
Trying to follow the structure of rnn_cell.py in the mxnet code.
"""

import mxnet as mx

import sockeye.constants as C
from sockeye.config import Config


import logging
logger = logging.getLogger(__name__)


#def get_gcn(input_dim: int, output_dim: int, 
#            tensor_dim: int, use_gcn_gating: bool, 
#            dropout: float, prefix: str):
def get_resgrn(config, prefix):
    resgrn = ResGRNCell(config.input_dim,
                        config.output_dim,
                        config.tensor_dim,
                        config.num_layers,
                        add_gate=config.add_gate,
                        prefix=prefix)
    return resgrn
   

# class GGRNParams(object):
#     """Container to hold GGRN variables.
#     Used for parameter sharing between layers/timesteps.

#     Parameters
#     ----------
#     prefix : str
#         All variables' name created by this container will
#         be prepended with prefix.
#     """

#     def __init__(self, prefix=''):
#         self._prefix = prefix
#         self._params = {}

#     def get(self, name, **kwargs):
#         """Get a variable with name or create a new one if missing.

#         Parameters
#         ----------
#         name : str
#             name of the variable
#         **kwargs :
#             more arguments that's passed to symbol.Variable
#         """
#         name = self._prefix + name
#         if name not in self._params:
#             self._params[name] = mx.sym.Variable(name, **kwargs)
#         return self._params[name]


class ResGRNConfig(Config):
    """
    GCN configuration.

    :param input_dim: Dimensionality for input vectors.
    :param output_dim: Dimensionality for output vectors.
    :param tensor_dim: Edge label space dimensionality.
    :param layers: Number of layers / unrolled timesteps.
    :param activation: Non-linear function used inside the GGRN updates.
    :param add_gate: Add edge-wise gating (Marcheggiani & Titov, 2017).
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tensor_dim: int,
                 num_layers: int,
                 activation: str = 'relu',
                 add_gate: bool = False) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tensor_dim = tensor_dim
        self.num_layers = num_layers
        self.activation = activation
        self.add_gate = add_gate
        
                 
class ResGRNCell(object):
    """Residual GRN cell
    """
    def __init__(self, input_dim, output_dim, tensor_dim, num_layers,
                 rank=256,
                 add_gate=False,
                 prefix='resgrn_', params=None, 
                 activation='relu',
                 dropout=0.0):
        #if params is None:
        #    params = GGRNParams(prefix)
        #    self._own_params = True
        #else:
        #    self._own_params = False
        self._own_params = True
        self._prefix = prefix
        self._params = params
        self._modified = False
        
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._tensor_dim = tensor_dim
        self._num_layers = num_layers
        self._rank = rank
        
        self._add_edge_gate = add_gate        
        self._activation = activation
        #self.reset()

        self._first_W = mx.symbol.Variable(self._prefix + '_first_weight',
                                           shape=(input_dim, output_dim))
        self._W = mx.symbol.Variable(self._prefix + '_weight',
                                     shape=(output_dim, rank))
        self._Wl = [mx.symbol.Variable(self._prefix + str(i) + '_edge_weight',
                                       shape=(rank, output_dim))
                    for i in range(tensor_dim)]
        self._bl = [mx.symbol.Variable(self._prefix + str(i) + '_edge_bias',
                                       shape=(output_dim,))
                    for i in range(tensor_dim)]
        # Edge gate parameters
        if self._add_edge_gate:
            self._edge_gate_W = [mx.symbol.Variable(self._prefix + str(i) + '_edge_gate_weight',
                                                    shape=(output_dim, 1))
                                 for i in range(tensor_dim)]
            self._edge_gate_b = [mx.symbol.Variable(self._prefix + str(i) + '_edge_gate_bias',
                                                    shape=(1, 1))
                                 for i in range(tensor_dim)]

    def convolve(self, adj, inputs, seq_len):
        """
        Apply one convolution per layer. This is where we add the residuals.
        A linear transformation is required in case the input dimensionality is
        different from GRN output dimensionality.
        """
        #outputs = self._single_convolve(adj, inputs, seq_len)
        #outputs = mx.symbol.FullyConnected(data=inputs, num_hidden=self._output_dim, flatten=True)
        outputs = mx.symbol.dot(inputs, self._first_W)
        #outputs = mx.symbol.concat(inputs, outputs)
        for i in range(self._num_layers - 1):
            outputs = self._single_convolve(adj, outputs, seq_len) + outputs
        #outputs = mx.symbol.concat(outputs, inputs)
        return outputs
            
    def _single_convolve(self, adj, inputs, seq_len):
        """
        IMPORTANT: when retrieving the original adj matrix for an
        edge label we add one to "i" because the edge ids stored
        in the matrix start at 1. 0 corresponds to lack of edges.
        """
        output_list = []
        for i in range(self._tensor_dim):
            # linear transformation
            Wi = self._Wl[i]
            Wi = mx.symbol.dot(self._W, Wi)
            bi = self._bl[i]            
            output = mx.symbol.dot(inputs, Wi)
            output = mx.symbol.broadcast_add(output, bi)
            # optional edge gating
            if self._add_edge_gate:
                edge_gate_Wi = self._edge_gate_W[i]
                edge_gate_bi = self._edge_gate_b[i]
                edge_gate_val = mx.symbol.dot(inputs, edge_gate_Wi)
                edge_gate_val = mx.symbol.broadcast_add(edge_gate_val, edge_gate_bi)
                edge_gate_val = mx.symbol.Activation(edge_gate_val, act_type='sigmoid')
                output = mx.symbol.broadcast_mul(output, edge_gate_val)
            # convolution
            label_id = i + 1
            mask = mx.symbol.ones_like(adj) * label_id
            adji = (mask == adj)
            #adji = mx.symbol.slice_axis(adj, axis=1, begin=i, end=i+1)
            #adji = mx.symbol.reshape(adji, shape=(-1, seq_len, seq_len))
            output = mx.symbol.batch_dot(adji, output)
            output = mx.symbol.expand_dims(output, axis=1)
            output_list.append(output)
        outputs = mx.symbol.concat(*output_list, dim=1)
        outputs = mx.symbol.sum(outputs, axis=1)
        final_output = mx.symbol.Activation(outputs, act_type=self._activation)
        #final_output = mx.symbol.Dropout(final_output, p=self._dropout)
        return final_output

    def reset(self):
        pass

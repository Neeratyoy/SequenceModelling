import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class RNNCell(nn.Module):
    """A vanilla RNN cell

    An RNN cell which takes two inputs: x and previous hidden state
    and outputs the current hidden state and output.

    The RNN cell can be used as components for the following designs:
    * One-to-one: https://stanford.edu/~shervine/images/rnn-one-to-one.png
    * One-to-many: https://stanford.edu/~shervine/images/rnn-one-to-many.png
    * Many-to-one: https://stanford.edu/~shervine/images/rnn-many-to-one.png
    * Many-to-many
        * Same: https://stanford.edu/~shervine/images/rnn-many-to-many-same.png
        * Different: https://stanford.edu/~shervine/images/rnn-many-to-many-different.png
    * Bidirectional RNNs: https://stanford.edu/~shervine/images/bidirectional-rnn.png
    * Deep RNNs: https://stanford.edu/~shervine/images/deep-rnn.png

    Parameters
    ==========
    input_dim: Dimension of input data
    output_dim: Dimension of outputs for each input
    hidden_dim: Size of hidden state

    """

    def __init__(self, input_dim, output_dim, hidden_dim, xavier_init=False):
        super().__init__()
        self.weights_hidden = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.weights_input = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.weights_output = nn.Parameter(torch.randn(hidden_dim, output_dim))
        self.bias_state = nn.Parameter(torch.randn(hidden_dim))
        self.bias_output = nn.Parameter(torch.randn(output_dim))
        self.g1 = nn.Sigmoid()
        self.g2 = nn.Tanh()

    def forward(self, x, hidden):
        if not hidden.requires_grad:
            raise ValueError("requires_grad needs to be True for hidden")
        timesteps = x.shape[0]
        output = torch.tensor([]).float()
        # Carries out the forward pass one timestep at a time
        for x_t in x:
            out, hidden = self.forward_pass(x_t.unsqueeze(0), hidden)
            output = torch.cat((output, out), dim=0)
        return output, hidden

    def forward_pass(self, x, hidden):
        # Design from:
        #    https://stanford.edu/~shervine/images/description-block-rnn.png
        #    https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

        # W_{ax}x^t
        input_actv = torch.matmul(x, self.weights_input)
        # W_{aa}a^{t-1}
        old_state_actv = torch.matmul(hidden, self.weights_hidden)
        # updated hidden state (new state)
        # g_1(W_{aa}a^{t-1} + W_{ax}x^t + b_a)
        hidden = self.g1(old_state_actv + input_actv + self.bias_state)

        # W_{ya}a^t + b_y
        output_actv = torch.matmul(hidden, self.weights_output)
        # g_2(W_{ya}a^t + b_y)
        output = self.g2(output_actv + self.bias_output)

        return output, hidden


class LSTMCell(nn.Module):
    """A vanilla LSTM cell

    An LSTM cell which takes two inputs: x and previous hidden state
    and outputs the current cell state and hidden state.
    Conditionally, an output cell can also be learnt where the final
    hidden state is reprojected to the required dimension.

    Parameters
    ==========
    input_dim: Dimension of input data
    hidden_dim: Size of hidden state
    output_dim: Dimension of outputs for each input

    """

    def __init__(self, input_dim, hidden_dim, layernorm=False):
        super().__init__()
        dim_size = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.layernorm = layernorm

        self.weights = nn.Parameter(torch.randn(dim_size, 4 * hidden_dim))
        # self.bias = torch.ones(4 * hidden_dim)
        self.bias = nn.Parameter(torch.randn(4 * hidden_dim))

        self.g1 = nn.Sigmoid()
        self.g2 = nn.Tanh()

        if self.layernorm:
            self.ln_gates = nn.LayerNorm(4 * hidden_dim)
            self.ln_candidate = nn.LayerNorm(hidden_dim)

        self.init_weights()

    def forward(self, x, hidden_state, cell_state, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Carries out the forward pass one timestep at a time
        timesteps = x.shape[0]
        output = torch.tensor([]).float().to(device)
        for x_t in x:
            cell_state, hidden_state = self.forward_pass(x_t.unsqueeze(0),
                                                         hidden_state, cell_state, device)
            output = torch.cat((output, hidden_state), dim=0)
        return output, (hidden_state, cell_state)

    def forward_pass(self, x, hidden_state, old_state, device=None):
        # Design and notations from:
        #    https://colah.github.io/posts/2015-08-Understanding-LSTMs/

        # h^{t-1}x^t
        concat_inputs = torch.cat((x, hidden_state), dim=2).to(device)

        gates = torch.matmul(concat_inputs, self.weights) + self.bias
        if self.layernorm:
            gates = self.ln_gates(gates)

        # Forget gate: $\Gamma_f = \sigma(W_f.[h_{t-1}, x_t] + b_f)$
        # Determines what(or how much) to throw away from old state
        forget_g = self.g1(gates[:, :, :self.hidden_dim])

        # Input gate: $\Gamma_i = \sigma(W_i.[h_{t-1}, x_t] + b_i)$
        # Decides which part of input is to be added
        input_g = self.g1(gates[:, :, self.hidden_dim:2 * self.hidden_dim])

        # Current candidate: $\widetilde{C_t} = tanh(W_C.[h_{t-1}, x_t] + b_C)$
        # Potential candidate to update the state with
        candidate_new = self.g2(gates[:, :, 2 * self.hidden_dim:3 * self.hidden_dim])

        # Output gate: $\Gamma_o = \sigma(W_o.[h_{t-1}, x_t] + b_o)$
        # Determines what to output based on current input and
        #    previous hidden state
        output_g = self.g1(gates[:, :, 3 * self.hidden_dim:])

        # New state: $C_t = f_t * C_{t-1} + i_t * \widetilde{C_t}$
        new_state = torch.mul(forget_g, old_state) + \
                    torch.mul(input_g, candidate_new)
        if self.layernorm:
            new_state = self.ln_candidate(new_state)

        # New hidden state/output: $h_t = \Gamma_o * tanh(C_t)$
        hidden_state = torch.mul(output_g, self.g2(new_state))

        return new_state, hidden_state

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=1, bidirectional=False, layernorm=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.layernorm = layernorm

        if self.layers < 1:
            raise ValueError("layers need to be > 1")
        self.model = []
        for i in range(self.layers):
            self.model.append(LSTMCell(input_dim, hidden_dim, layernorm))
        self.model = nn.ModuleList(self.model)
        if self.bidirectional:
            self.model_rev = []
            for i in range(self.layers):
                self.model_rev.append(LSTMCell(input_dim, hidden_dim, layernorm))
            self.model_rev = nn.ModuleList(self.model_rev)

    def forward(self, x, hidden_state, cell_state, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        seq_length = x.shape[0]
        # Left-to-right pass
        # index of state is equivalent to index of layer in LSTM stack
        hidden_states = torch.cat(tuple(hidden_state.clone().unsqueeze(0)
                                        for i in range(self.layers)), dim=0)
        cell_states = torch.cat(tuple(cell_state.clone().unsqueeze(0)
                                      for i in range(self.layers)), dim=0)
        output = torch.tensor([], requires_grad=True).to(device)
        # forward pass for one cell at a time
        for j in range(self.layers):
            output, (hidden_states[j], cell_states[j]) = self.model[j](x, hidden_states[j].clone(),
                                                                  cell_states[j].clone(),
                                                                  device)
        hidden_states = hidden_states.view(self.layers, hidden_states.shape[-2],
                                           hidden_states.shape[-1])
        cell_states = cell_states.view(self.layers, cell_states.shape[-2],
                                         cell_states.shape[-1])

        # Right-to-left pass
        if self.bidirectional:
            # index of state is equivalent to index of layer in LSTM stack
            hidden_states_rev = torch.cat(tuple(hidden_state.clone().unsqueeze(0)
                                                for i in range(self.layers)), dim=0)
            cell_states_rev = torch.cat(tuple(cell_state.clone().unsqueeze(0)
                                              for i in range(self.layers)), dim=0)
            output_rev = torch.tensor([], requires_grad=True).to(device)
            # forward pass for one cell at a time
            for j in range(self.layers):
                output_rev, (hidden_states_rev[j], cell_states_rev[j]) = self.model_rev[j](x,
                                                                        hidden_states_rev[j].clone(),
                                                                        cell_states_rev[j].clone())
            # last_layer_output_rev = o_rev
            hidden_states_rev = hidden_states_rev.view(self.layers, hidden_states_rev.shape[-2],
                                                       hidden_states_rev.shape[-1])
            cell_states_rev = cell_states_rev.view(self.layers, cell_states_rev.shape[-2],
                                                   cell_states_rev.shape[-1])
            # concatenating tensors
            ## creating tensors as expected in
            ## here: https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
            ## follows the unpacked structure when bidirectional
            hidden_states = torch.cat((hidden_states.unsqueeze(1),
                                       hidden_states_rev.unsqueeze(1)), dim=1)
            cell_states = torch.cat((cell_states.unsqueeze(1),
                                     cell_states_rev.unsqueeze(1)), dim=1)
            output = torch.cat((output.unsqueeze(2),
                                output_rev.unsqueeze(2)), dim=2)

        return output, (hidden_states, cell_states)



# Below class was the inital bare-bones implementation
#
# class LSTMCell(nn.Module):
#     """A vanilla LSTM cell
#
#     An LSTM cell which takes two inputs: x and previous hidden state
#     and outputs the current cell state and hidden state.
#     Conditionally, an output cell can also be learnt where the final
#     hidden state is reprojected to the required dimension.
#
#     Parameters
#     ==========
#     input_dim: Dimension of input data
#     hidden_dim: Size of hidden state
#     output_dim: Dimension of outputs for each input
#
#     """
#
#     def __init__(self, input_dim, hidden_dim, xavier_init=False):
#         super().__init__()
#         dim_size = input_dim + hidden_dim
#         self.weights_forget = nn.Parameter(torch.randn(dim_size, hidden_dim))
#         self.bias_forget = nn.Parameter(torch.randn(hidden_dim))
#         self.weights_input = nn.Parameter(torch.randn(dim_size, hidden_dim))
#         self.bias_input = nn.Parameter(torch.randn(hidden_dim))
#         self.weights_candidate = nn.Parameter(torch.randn(dim_size, hidden_dim))
#         self.bias_candidate = nn.Parameter(torch.randn(hidden_dim))
#         self.weights_output = nn.Parameter(torch.randn(dim_size, hidden_dim))
#         self.bias_output = nn.Parameter(torch.randn(hidden_dim))
#         self.g1 = nn.Sigmoid()
#         self.g2 = nn.Tanh()
#         if xavier_init:
#             self.initialize_with_Xavier()
#
#     def forward(self, x, hidden_state, cell_state):
#         # Carries out the forward pass one timestep at a time
#         timesteps = x.shape[0]
#         output = torch.tensor([]).float()
#         for x_t in x:
#             cell_state, hidden_state = self.forward_pass(x_t.unsqueeze(0),
#                                                          hidden_state, cell_state)
#             output = torch.cat((output, hidden_state), dim=0)
#         return output, (cell_state, hidden_state)
#
#     def forward_pass(self, x, hidden_state, old_state):
#         # Design and notations from:
#         #    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
#
#         # h^{t-1}x^t
#         concat_inputs = torch.cat((x, hidden_state), dim=2)
#
#         # Forget gate: $\Gamma_f = \sigma(W_f.[h_{t-1}, x_t] + b_f)$
#         # Determines what(or how much) to throw away from old state
#         forget_g = self.g1(torch.matmul(concat_inputs, self.weights_forget) +
#                            self.bias_forget)
#
#         # Input gate: $\Gamma_i = \sigma(W_i.[h_{t-1}, x_t] + b_i)$
#         # Decides which part of input is to be added
#         input_g = self.g1(torch.matmul(concat_inputs, self.weights_input) +
#                           self.bias_input)
#
#         # Current candidate: $\widetilde{C_t} = tanh(W_C.[h_{t-1}, x_t] + b_C)$
#         # Potential candidate to update the state with
#         candidate_new = self.g2(torch.matmul(concat_inputs, self.weights_candidate) +
#                                 self.bias_candidate)
#
#         # Output gate: $\Gamma_o = \sigma(W_o.[h_{t-1}, x_t] + b_o)$
#         # Determines what to output based on current input and
#         #    previous hidden state
#         output_g = self.g1(torch.matmul(concat_inputs, self.weights_output) +
#                            self.bias_output)
#
#         # New state: $C_t = f_t * C_{t-1} + i_t * \widetilde{C_t}$
#         new_state = torch.mul(forget_g, old_state) + \
#                     torch.mul(input_g, candidate_new)
#         # New hidden state/output: $h_t = \Gamma_o * tanh(C_t)$
#         hidden_state = torch.mul(output_g, self.g2(new_state))
#
#         return new_state, hidden_state
#
#     def initialize_with_Xavier(self):  # , n_var, dim_size, hidden_dim, output_dim):
#         for p in self.parameters():
#             if p.data.ndimension() >= 2:
#                 nn.init.xavier_uniform_(p.data)

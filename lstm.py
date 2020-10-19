import numpy as np
from math_utils import sigmoid, dsigmoid, tanh, dtanh

class LSTM:
    def __init__(self, hidden_size, input_dim, batch_size, lr):
        self.hidden_size = hidden_size
        self.input_dim = input_dim
        self.lr = lr
        self.batch_size = batch_size

        self.upper_matrix_init_value = 0.1
        self.lower_matrix_init_value = -0.1

        self.init_weights()
        self.init_biases()

    def get_matrix(self, num_rows, num_cols):
        return np.random.rand(num_rows, num_cols) * (self.upper_matrix_init_value - self.lower_matrix_init_value) + self.lower_matrix_init_value

    def init_weights(self):
        '''
            f: forget
            i: input
            o: output
            c: cell (memory)

            U: Weights on input
            W: Weights on hidden states
        '''
        # Weights on input
        self.U_f = self.get_matrix(self.hidden_size, self.input_dim)
        self.U_i = self.get_matrix(self.hidden_size, self.input_dim)
        self.U_o = self.get_matrix(self.hidden_size, self.input_dim)
        self.U_c = self.get_matrix(self.hidden_size, self.input_dim)

        # Weights on hidden state
        self.W_f = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_i = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_o = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_c = self.get_matrix(self.hidden_size, self.hidden_size)

    def init_weights_derivatives(self):
        # Derivatives of weights on input
        self.U_f_diff = np.zeros_like(self.U_f)
        self.U_i_diff = np.zeros_like(self.U_i)
        self.U_o_diff = np.zeros_like(self.U_o)
        self.U_c_diff = np.zeros_like(self.U_c)

        # Derivatives of weights on hidden state
        self.W_f_diff = np.zeros_like(self.W_f)
        self.W_i_diff = np.zeros_like(self.W_i)
        self.W_o_diff = np.zeros_like(self.W_o)
        self.W_c_diff = np.zeros_like(self.W_c)

    def init_biases(self):
        # Biases on input
        self.B_fx = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_ix = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_ox = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_cx = self.get_matrix(self.hidden_size, self.batch_size)

        # Biases on hidden state
        self.B_fh = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_ih = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_oh = self.get_matrix(self.hidden_size, self.batch_size)
        self.B_ch = self.get_matrix(self.hidden_size, self.batch_size)

    def init_biases_derivatives(self):
        # Derivatives of biases on input
        self.B_fx_diff = np.zeros_like(self.B_fx)
        self.B_ox_diff = np.zeros_like(self.B_ox)
        self.B_ix_diff = np.zeros_like(self.B_ix)
        self.B_cx_diff = np.zeros_like(self.B_cx)

        # Derivatives of biases on hidden state
        self.B_fh_diff = np.zeros_like(self.B_fh)
        self.B_ih_diff = np.zeros_like(self.B_ih)
        self.B_oh_diff = np.zeros_like(self.B_oh)
        self.B_ch_diff = np.zeros_like(self.B_ch)

    def update_weights(self):
        '''
            Helper method to update weights
        '''
        self.U_f -= self.U_f_diff * self.lr 
        self.U_i -= self.U_i_diff * self.lr
        self.U_o -= self.U_o_diff * self.lr
        self.U_c -= self.U_c_diff * self.lr

        self.W_f -= self.W_f_diff * self.lr 
        self.W_i -= self.W_i_diff * self.lr 
        self.W_o -= self.W_o_diff * self.lr 
        self.W_c -= self.W_c_diff * self.lr 

        # Reset weight derivatives to 0
        self.init_weights_derivatives()

    def update_biases(self):
        '''
            Helper method to update biases
        '''
        self.B_fx -= self.B_fx_diff * self.lr
        self.B_ix -= self.B_ix_diff * self.lr
        self.B_ox -= self.B_ox_diff * self.lr
        self.B_cx -= self.B_cx_diff * self.lr

        self.B_fh -= self.B_fh_diff * self.lr
        self.B_ih -= self.B_ih_diff * self.lr
        self.B_oh -= self.B_oh_diff * self.lr
        self.B_ch -= self.B_ch_diff * self.lr

        # Reset bias derivatives to 0
        self.init_biases_derivatives()

    def forget_gate(self, x, h_prev, cell_prev):
        forget_x      = np.dot(self.U_f, x) + self.B_fx
        forget_h_prev = np.dot(self.W_f, h_prev) + self.B_fh

        return sigmoid(forget_x + forget_h_prev)
    
    def input_gate(self, x, h_prev, cell_prev):
        input_x      = np.dot(self.U_i, x) + self.B_ix
        input_h_prev = np.dot(self.W_i, h_prev) + self.B_ih

        return sigmoid(input_x + input_h_prev)

    def output_gate(self, x, h_prev, cell_prev):
        out_x      = np.dot(self.U_o, x) + self.B_ox
        out_h_prev = np.dot(self.W_o, h_prev) + self.B_oh

        return sigmoid(out_x + out_h_prev)

    def candidate(self, x, h_prev, cell_prev):
        candidate_x      = np.dot(self.U_c, x) + self.B_cx
        candidate_h_prev = np.dot(self.W_c, h_prev) + self.B_ch

        return tanh(candidate_x + candidate_h_prev)

    def update_cell_state(self, forget_gate, input_gate, candidate, cell_prev):
        a = candidate * input_gate
        b = input_gate * candidate

        return a + b

    def forward_pass(self, x, h_prev, cell_prev):
        '''
            At the current time step, calculate the values for all the gates as well as
            calculate the cell and hidden states.

            Parameters:
            x -- The input at the current time step t
            h_prev -- The hidden state from the previous time step t-1
            c_prev -- The cell (memory) state from the previous time step t-1
        '''

        # Gates
        forget_gate = self.forget_gate(x, h_prev, cell_prev)    # [H x B]
        input_gate = self.input_gate(x, h_prev, cell_prev)      # [H x B]
        out_gate = self.output_gate(x, h_prev, cell_prev)       # [H x B]

        # States
        candidate = self.candidate(x, h_prev, cell_prev)        # [H x B]
        cell_current = self.update_cell_state(forget_gate, input_gate, candidate, cell_prev) # [H x B]
        h_current = out_gate * tanh(cell_current)               # [H x B]

        return input_gate, forget_gate, out_gate, candidate, cell_current, h_current

    def backward_pass(forward_pass_outputs):
        input_gate, forget_gate, out_gate, candidate, cell_current, h_current = forward_pass_outputs
        


H = 3
D = 2
B = 1

lstm = LSTM(hidden_size=H, input_dim=D, batch_size=B, lr = 1)

x = np.random.rand(D, B)
h_prev = np.zeros((H, B))
cell_prev = np.zeros((H, B))

lstm.forward_pass(x, h_prev, cell_prev)











# https://github.com/keras-team/keras/issues/3088
# https://towardsdatascience.com/examining-the-weight-and-bias-of-lstm-in-tensorflow-2-5576049a91fa
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/
# https://www.quora.com/In-LSTM-how-do-you-figure-out-what-size-the-weights-are-supposed-to-be
# https://github.com/nicklashansen/rnn_lstm_from_scratch/blob/master/RNN_LSTM_from_scratch.ipynb
# https://github.com/nicodjimenez/lstm/blob/master/test.py
# https://github.com/nicodjimenez/lstm
# http://nicodjimenez.github.io/2014/08/08/lstm.html
# https://towardsdatascience.com/the-lstm-reference-card-6163ca98ae87
# https://gist.github.com/conditg/47eb195eb1d5b80ea299c567c8d0f3bf
# https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras
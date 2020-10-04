import numpy as np
from math_utils import sigmoid

class LSTM:
    def __init__(self, hidden_size, input_dim):
        self.hidden_size = hidden_size
        self.input_dim = input_dim
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
        '''
        # Weights on input
        self.W_fx = self.get_matrix(self.hidden_size, self.input_dim)
        self.W_ix = self.get_matrix(self.hidden_size, self.input_dim)
        self.W_ox = self.get_matrix(self.hidden_size, self.input_dim)
        self.W_cx = self.get_matrix(self.hidden_size, self.input_dim)

        # Weights on hidden state
        self.W_fh = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_ih = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_oh = self.get_matrix(self.hidden_size, self.hidden_size)
        self.W_ch = self.get_matrix(self.hidden_size, self.hidden_size)

    def init_biases(self):
        # Biases on input
        self.B_fx = self.get_matrix(self.hidden_size, 1)
        self.B_ix = self.get_matrix(self.hidden_size, 1)
        self.B_ox = self.get_matrix(self.hidden_size, 1)
        self.B_cx = self.get_matrix(self.hidden_size, 1)

        # Biases on hidden state
        self.B_fh = self.get_matrix(self.hidden_size, 1)
        self.B_ih = self.get_matrix(self.hidden_size, 1)
        self.B_oh = self.get_matrix(self.hidden_size, 1)
        self.B_ch = self.get_matrix(self.hidden_size, 1)

    def forward_pass(self, x, h_prev, cell_prev):
        '''
            Parameters:
            x -- The input at the current time step t
            h_prev -- The hidden state from the previous time step t-1
            c_prev -- The cell (memory) state from the previous time step t-1
        '''

        # Forget gate
        forget_x      = np.dot(self.W_fx, x) + self.B_fx
        forget_h_prev = np.dot(self.W_fh, h_prev) + self.B_fh
        forget_gate   = sigmoid(forget_x + forget_h_prev)

        # Input gate
        input_x      = np.dot(self.W_ix, x) + self.B_ix
        input_h_prev = np.dot(self.W_ih, h_prev) + self.B_ih
        input_gate   = sigmoid(input_x + input_h_prev)

        # Candidate cell state
        candidate_x      = np.dot(self.W_cx, x) + self.B_cx
        candidate_h_prev = np.dot(self.W_ch, h_prev) + self.B_ch
        candidate        = np.tanh(candidate_x + candidate_h_prev)

        # Output gate
        out_x    = np.dot(self.W_ox, x) + self.B_ox
        out_h    = np.dot(self.W_oh, h_prev) + self.B_oh
        out_gate = sigmoid(out_x + out_h)

        # Current cell state
        cell_current = np.multiply(cell_prev, forget_gate) + np.multiply(input_gate, candidate)

        # Current hidden state
        h_current = np.multiply(out_gate, np.tanh(cell_current))

        return cell_current, h_current

    def backward_pass():
        

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
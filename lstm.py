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





    def get_matrix(self, x_dim, y_dim):
        return np.random.rand(x_dim, y_dim) * (self.upper_matrix_init_value - self.lower_matrix_init_value) + self.lower_matrix_init_value



lstm = LSTM(5,4)









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
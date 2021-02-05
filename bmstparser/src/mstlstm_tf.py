import numpy as np
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class MSTParserLSTMTF:

    def __init__(self):
        # LSTM Network architecture
        self.time_steps = 14            # Number of words in a sentence
        self.number_features = 100
        self.hidden_state_size = 125
        self.number_features_2 = self.hidden_state_size * 2

        # Input data
        self.sample_size = 1  # batch size

        # First LSTM Layer
        inputs_1 = Input(shape=(self.time_steps, self.number_features))
        hidden_states, hidden_state, cell_state = \
            LSTM(self.hidden_state_size, return_sequences=True, return_state=True)(inputs_1)
        self.lstm_for_1 = Model(inputs=inputs_1, outputs=[hidden_states, hidden_state, cell_state])
        self.lstm_back_1 = Model(inputs=inputs_1, outputs=[hidden_states, hidden_state, cell_state])

        # Second LSTM Layer
        inputs_2 = Input(shape=(self.time_steps, self.number_features_2))
        hidden_states_2, hidden_state_2, cell_state_2 = \
            LSTM(self.hidden_state_size, return_sequences=True, return_state=True)(inputs_2)
        self.lstm_for_2 = Model(inputs=inputs_2, outputs=[hidden_states_2, hidden_state_2, cell_state_2])
        self.lstm_back_2 = Model(inputs=inputs_2, outputs=[hidden_states_2, hidden_state_2, cell_state_2])

        # Variables required for prediction
        # TODO: Check if this variables are really required at this level
        self.hid_for_1 = ()
        self.hid_back_1 = ()
        self.hid_for_2 = ()
        self.hid_back_2 = ()

    def forward(self):
        vec_for = self.get_input(self.number_features)
        vec_back = np.flip(vec_for, 1)

        output_for_1, self.hid_for_1 = self.get_lstm_output(self.lstm_for_1, vec_for)
        output_back_1, self.hid_back_1 = self.get_lstm_output(self.lstm_back_1, vec_back)

        vec_for_2 = self.get_input(self.number_features_2)
        vec_back_2 = np.flip(vec_for_2, 1)

        output_for_2, self.hid_for_2 = self.get_lstm_output(self.lstm_for_2, vec_for_2)
        output_back_2, self.hid_back_2 = self.get_lstm_output(self.lstm_back_2, vec_back_2)

    def get_input(self, number_features):
        # Since in Pytorch is [14, 1, 100] in Tensorflow should be [1, 14, 100]
        random_input = np.random.rand(self.time_steps, number_features)
        input_sequence = random_input.reshape((self.sample_size, self.time_steps, number_features))
        print("Input Sequence: " + str(input_sequence))
        return input_sequence

    @staticmethod
    def get_lstm_output(lstm_model, input_sequence):
        lstm_model.summary()
        output = lstm_model.predict(input_sequence)
        hidden_states, hidden_state, cell_state = output[0], output[1], output[2]
        return hidden_states, (hidden_state, cell_state)


if __name__ == '__main__':
    lstm_net = MSTParserLSTMTF()
    lstm_net.forward()

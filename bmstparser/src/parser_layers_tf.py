import tensorflow as tf
from tensorflow.keras.layers import Embedding
import utils_tf


class EmbeddingsModule(tf.keras.layers.Layer):

    def __init__(self, vocab_size, wdims):
        super().__init__(name="EmbeddingsModule")
        self.wdims = wdims

        self.wlookup = Embedding(vocab_size, self.wdims, name='embedding_vocab',
                                 embeddings_initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))

    def call(self, inputs):
        # Forward pass
        word_vec = self.wlookup(inputs) if self.wdims > 0 else None
        return word_vec


class BiLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__(name="BiLSTMModule")
        self.blockLstm = FirstBlockLSTMModule(lstm_dims)
        self.nextBlockLstm = NextBlockLSTM(lstm_dims)
        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs):

        block_lstm1_output = self.blockLstm(inputs)
        block_lstm2_output = self.nextBlockLstm(block_lstm1_output)
        return block_lstm2_output


class FirstBlockLSTMModule(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__(name="FirstBlockLSTMModule")
        self.initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform
        self.sample_size = 1
        self.ini_cell_state = tf.zeros(shape=[self.sample_size, lstm_dims], name='c_first_lstm')
        self.ini_hidden_state = tf.zeros(shape=[self.sample_size, lstm_dims], name='h_first_lstm')

        self.bias = tf.zeros(shape=[lstm_dims * 4], name='b_first_lstm')
        self.lstm_dims = lstm_dims

    def build(self, input_shape):
        self.input_size = input_shape[0].dims[2]
        values = self.initializer(shape=[self.input_size + self.lstm_dims, self.lstm_dims * 4])
        self.weight_matrix = tf.Variable(values, name='w_first_lstm')

        values = self.initializer(shape=[self.lstm_dims])
        self.weight_input_gate = tf.Variable(values, name='wig_first_lstm')
        self.weight_forget_gate = tf.Variable(values, name='wfg_first_lstm')
        self.weight_output_gate = tf.Variable(values, name='wog_first_lstm')

    def call(self, inputs):
        time_steps = inputs[0].shape.dims[1]

        vec_for = tf.reshape(inputs[0], shape=[time_steps, self.sample_size, self.input_size])
        vec_back = tf.reshape(inputs[1], shape=[time_steps, self.sample_size, self.input_size])

        block_lstm_for_1 = self.get_lstm_output(vec_for, time_steps)
        block_lstm_back_1 = self.get_lstm_output(vec_back, time_steps)

        res_for_1 = tf.reshape(block_lstm_for_1.h, shape=[self.sample_size, time_steps, self.lstm_dims])
        res_back_1 = tf.reshape(block_lstm_back_1.h, shape=[self.sample_size, time_steps, self.lstm_dims])

        vec_cat = self.concatenate_layers(res_for_1[0], res_back_1[0], time_steps)
        vec_for_2 = tf.reshape(tf.concat(vec_cat, 0), shape=(self.sample_size, time_steps, -1))
        vec_back_2 = tf.reshape(tf.concat(list(reversed(vec_cat)), 0), shape=(self.sample_size, time_steps, -1))

        return [vec_for_2, vec_back_2]

    def get_lstm_output(self, input_sequence, time_steps):
        block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=input_sequence, cs_prev=self.ini_cell_state,
                                          h_prev=self.ini_hidden_state, w=self.weight_matrix,
                                          wci=self.weight_input_gate, wcf=self.weight_forget_gate,
                                          wco=self.weight_output_gate, b=self.bias)
        return block_lstm

    @staticmethod
    def concatenate_layers(array1, array2, num_vec):
        concat_size = array1.shape[1] + array2.shape[1]
        concat_result = [tf.reshape(tf.concat([array1[i], array2[num_vec - i - 1]], 0), shape=(1, concat_size))
                         for i in range(num_vec)]
        return concat_result


class NextBlockLSTM(tf.keras.layers.Layer):

    def __init__(self, lstm_dims):
        super().__init__(name="NextBlockLSTM")
        self.initializer = tf.keras.initializers.GlorotUniform()  # Xavier uniform
        self.sample_size = 1
        self.ini_cell_state = tf.zeros(shape=[self.sample_size, lstm_dims], name='c_next_lstm')
        self.ini_hidden_state = tf.zeros(shape=[self.sample_size, lstm_dims], name='h_next_lstm')

        self.bias = tf.zeros(shape=[lstm_dims * 4], name='b_next_lstm')
        self.lstm_dims = lstm_dims

    def build(self, input_shape):
        self.input_size = input_shape[0].dims[2]
        values = self.initializer(shape=[self.input_size + self.lstm_dims, self.lstm_dims * 4])
        self.weight_matrix = tf.Variable(values, name='w_next_lstm')

        values = self.initializer(shape=[self.lstm_dims])
        self.weight_input_gate = tf.Variable(values, name='wig_next_lstm')
        self.weight_forget_gate = tf.Variable(values, name='wfg_next_lstm')
        self.weight_output_gate = tf.Variable(values, name='wog_next_lstm')

    def call(self, inputs):
        # Forward pass
        time_steps = inputs[0].shape.dims[1]

        vec_for_2 = tf.reshape(inputs[0], shape=[time_steps, self.sample_size, self.input_size])
        vec_back_2 = tf.reshape(inputs[1], shape=[time_steps, self.sample_size, self.input_size])

        block_lstm_for_2 = self.get_lstm_output(vec_for_2, time_steps)
        block_lstm_back_2 = self.get_lstm_output(vec_back_2, time_steps)

        res_for_2 = tf.reshape(block_lstm_for_2.h, shape=[self.sample_size, time_steps, self.lstm_dims])
        res_back_2 = tf.reshape(block_lstm_back_2.h, shape=[self.sample_size, time_steps, self.lstm_dims])

        return [res_for_2, res_back_2]

    def get_lstm_output(self, input_sequence, time_steps):
        block_lstm = tf.raw_ops.BlockLSTM(seq_len_max=time_steps, x=input_sequence, cs_prev=self.ini_cell_state,
                                          h_prev=self.ini_hidden_state, w=self.weight_matrix,
                                          wci=self.weight_input_gate, wcf=self.weight_forget_gate,
                                          wco=self.weight_output_gate, b=self.bias)
        return block_lstm


class ConcatHeadModule(tf.keras.layers.Layer):

    def __init__(self, ldims, hidden_units, hidden2_units, activation):
        super().__init__("ConcatHeadModule")

        self.activation = activation
        self.hidden2_units = hidden2_units
        self.hidLayerFOH = utils_tf.Parameter((ldims * 2, hidden_units), 'hidLayerFOH')
        self.hidLayerFOM = utils_tf.Parameter((ldims * 2, hidden_units), 'hidLayerFOM')
        self.hidBias = utils_tf.Parameter(hidden_units, 'hidBias')
        self.catBias = utils_tf.Parameter(hidden_units * 2, 'catBias')

        if self.hidden2_units:
            self.hid2Layer = utils_tf.Parameter((hidden_units * 2, hidden2_units), 'hid2Layer')
            self.hid2Bias = utils_tf.Parameter(hidden2_units, 'hid2Bias')

        self.outLayer = utils_tf.Parameter((hidden2_units if hidden2_units > 0 else hidden_units, 1), 'outLayer')
        self.outBias = utils_tf.Parameter(1, 'outBias')

        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs):
        # Forward pass
        scores, exprs = self.__evaluate(inputs)

        return [scores, exprs]

    def __evaluate(self, inputs):

        def transform_tensor(tensor_list):
            return list(map(lambda tensor: tensor[0, 0], tensor_list))

        head_vector = []  # TODO: Check if this array must be a Tensor variable to save it in the model
        for index in range(len(inputs)):
            lstms_0 = inputs[index][0]
            lstms_1 = inputs[index][1]
            concatenated_lstm = tf.reshape(utils_tf.concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
            headfov = tf.matmul(concatenated_lstm, self.hidLayerFOH)
            modfov = tf.matmul(concatenated_lstm, self.hidLayerFOM)
            head_vector.append([headfov, modfov])

        exprs = [[self.__getExpr(head_vector, i, j) for j in range(len(head_vector))] for i in range(len(head_vector))]

        output_tensor = [[output for output in exprsRow] for exprsRow in exprs]
        # scores = np.array([convert_to_numpy(output) for output in output_tensor])
        scores = tf.stack([transform_tensor(output) for output in output_tensor])

        return scores, exprs

    def __getExpr(self, head_vector, i, j):
        headfov = head_vector[i][0]
        modfov = head_vector[j][1]
        if self.hidden2_units > 0:
            concatenated_result = utils_tf.concatenate_tensors([headfov, modfov])
            activation_result = self.activation(concatenated_result + self.catBias)
            matmul_result = tf.matmul(activation_result, self.hid2Layer)
            next_activation_result = self.activation(self.hid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.outLayer) + self.outBias
        else:
            activation_result = self.activation(headfov + modfov + self.hidBias)
            output = tf.matmul(activation_result, self.outLayer) + self.outBias

        return output


class ConcatRelationModule(tf.keras.layers.Layer):

    def __init__(self, relations_size, ldims, hidden_units, hidden2_units, activation):
        super().__init__(name="ConcatRelationModule")

        self.activation = activation
        self.hidden2_units = hidden2_units
        self.rhidLayerFOH = utils_tf.Parameter((2 * ldims, hidden_units), 'rhidLayerFOH')
        self.rhidLayerFOM = utils_tf.Parameter((2 * ldims, hidden_units), 'rhidLayerFOM')
        self.rhidBias = utils_tf.Parameter(hidden_units, 'rhidBias')
        self.rcatBias = utils_tf.Parameter(hidden_units * 2, 'rcatBias')

        if self.hidden2_units:
            self.rhid2Layer = utils_tf.Parameter((hidden_units * 2, hidden2_units), 'rhid2Layer')
            self.rhid2Bias = utils_tf.Parameter(hidden2_units, 'rhid2Bias')

        self.routLayer = utils_tf.Parameter((hidden2_units if hidden2_units > 0 else hidden_units, relations_size),
                                            'routLayer')
        self.routBias = utils_tf.Parameter(relations_size, 'routBias')

        self.__sentence = None

    def set_sentence(self, sentence):
        self.__sentence = sentence

    def call(self, inputs, training):
        # Forwad pass
        # TODO: Add gold to the inputs
        rscores_list = []
        if self.__sentence is None:
            return rscores_list  # TODO: Remove sentence dependency to save relations variables on model
        gold = [entry.parent_id for entry in self.__sentence]
        for modifier, head in enumerate(gold[1:]):
            lstms_0 = inputs[head][0]
            lstms_1 = inputs[modifier + 1][1]

            if self.__sentence[head].rheadfov is None:
                concatenated_lstm = tf.reshape(utils_tf.concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
                self.__sentence[head].rheadfov = tf.matmul(concatenated_lstm, self.rhidLayerFOH)

            if self.__sentence[modifier + 1].modfov is None:
                concatenated_lstm = tf.reshape(utils_tf.concatenate_tensors([lstms_0, lstms_1]), shape=(1, -1))
                self.__sentence[modifier + 1].modfov = tf.matmul(concatenated_lstm, self.rhidLayerFOM)

            rscores = self.__evaluateLabel(self.__sentence[head].rheadfov, self.__sentence[modifier + 1].modfov)
            rscores_list.append(rscores)

        return rscores_list

    def __evaluateLabel(self, rheadfov, rmodfov):

        if self.hidden2_units > 0:
            concatenated_result = utils_tf.concatenate_tensors([rheadfov, rmodfov])
            activation_result = self.activation(concatenated_result + self.rcatBias)
            matmul_result = tf.matmul(activation_result, self.rhid2Layer)
            next_activation_result = self.activation(self.rhid2Bias + matmul_result)
            output = tf.matmul(next_activation_result, self.routLayer) + self.routBias
        else:
            activation_result = self.activation(rheadfov + rmodfov + self.rhidBias)
            output = tf.matmul(activation_result, self.routLayer) + self.routBias

        return output[0]

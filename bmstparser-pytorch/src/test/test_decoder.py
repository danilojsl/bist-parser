import tensorflow as tf
import decoder_v2_tf as decoder


def test_parser():
    mocked_score_matrix_5x5 = tf.constant([[1, 2, 3, 4, 5],
                                           [6, 7, 8, 9, 10],
                                           [1, 12, 12, 14, 15],
                                           [16, 17, 18, 19, 20],
                                           [21, 22, 23, 24, 25]])

    mocked_score_matrix_3x3 = tf.constant([[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]])

    decoder.parse_proj(mocked_score_matrix_3x3)


def validate_first_edge_case(s, t):
    print("On First Edge Case")
    # complete[s, s:t, 1] traversing 3th dimension with index 1 on complete matrix
    result = matrix_np[s, s:t, 1]
    result_tf = tf.reshape(tf.slice(matrix, begin=[s, s, 1], size=[1, t-s, 1]), shape=(t-s))
    print(f"Results with s={s}, t={t}")
    print(result)
    print(result_tf)
    assert result[0] == result_tf[0]


def validate_second_edge_case(s, t):
    print("On Second Edge Case")
    # complete[(s + 1):(t + 1), t, 0] traversing 3th dimension with index 0 on complete matrix
    result = matrix_np[(s + 1):(t + 1), t, 0]
    index = (t + 1) - (s + 1)
    result_tf = tf.reshape(tf.slice(matrix, begin=[s + 1, t, 0], size=[index, 1, 1]), shape=index)
    print(f"Results with s={s}, t={t}")
    print(result)
    print(result_tf)
    assert result[0] == result_tf[0]


def validate_third_edge_case(s, t):
    print("On Third Edge Case")
    # scores[t, s] traversing score matrix
    result = scores_np[t, s]
    result_tf = tf.reshape(tf.slice(scores, begin=[t, s], size=[1, 1]), shape=1)
    print(f"Results with s={s}, t={t}")
    print(result)
    print(result_tf)
    assert result == result_tf[0]


def validate_forth_edge_case(s, t):
    print("On Forth Edge Case")
    # complete[s, s:t, 0] traversing 3th dimension with index 0 on complete matrix
    result = matrix_np[s, s:t, 0]
    result_tf = tf.reshape(tf.slice(matrix, begin=[s, s, 0], size=[1, t-s, 1]), shape=(t-s))
    print(f"Results with s={s}, t={t}")
    print(result)
    print(result_tf)
    assert result[0] == result_tf[0]


def validate_fifth_edge_case(s, t):
    print("On Fifth Edge Case")
    # complete[(s + 1):(t + 1), t, 1] traversing 3th dimension with index 1 on complete matrix
    result = matrix_np[(s + 1):(t + 1), t, 1]
    index = (t + 1) - (s + 1)
    result_tf = tf.reshape(tf.slice(matrix, begin=[s + 1, t, 1], size=[index, 1, 1]), shape=index)
    print(f"Results with s={s}, t={t}")
    print(result)
    print(result_tf)
    assert result[0] == result_tf[0]


if __name__ == '__main__':
    scores = tf.constant([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])
    scores_np = scores.numpy()

    matrix = tf.constant([[[1, 2], [3, 4], [5, 6]],
                          [[7, 8], [9, 10], [11, 12]],
                          [[13, 14], [15, 16], [17, 18]]])

    matrix_np = matrix.numpy()

    print("Playing with dimensions")
    rows_number = int(matrix.shape.dims[0])
    for k_index in range(1, rows_number):
        for s_index in range(rows_number - k_index):
            t_index = s_index + k_index
            validate_first_edge_case(s_index, t_index)
            # validate_second_edge_case(s_index, t_index)
            # validate_third_edge_case(s_index, t_index)
            # validate_forth_edge_case(s_index, t_index)
            # validate_fifth_edge_case(s_index, t_index)


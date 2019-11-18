import tensorflow as tf


def residual_block(input_layer, kernel_size, output_channel, use_batch_normalize=True, is_training=True):
    """
    Defines a residual block in AlphaGo zero.
    :param input_layer: 4D tensor
    :param kernel_size: tuple, e.g. (3, 3)
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param use_batch_normalize: bool,
    :param is_training: bool,
    :return: 4D tensor.
    """
    output = input_layer

    with tf.variable_scope('conv-1_in_residual'):
        output = tf.layers.conv2d(inputs=output, filters=output_channel,
                                  kernel_size=kernel_size, strides=(1, 1),
                                  padding='SAME')
        if use_batch_normalize:
            output = tf.layers.batch_normalization(output, training=is_training)
        output = tf.nn.relu(output)

    with tf.variable_scope('conv-2_in_residual'):
        output = tf.layers.conv2d(inputs=output, filters=output_channel,
                                  kernel_size=kernel_size, strides=(1, 1),
                                  padding='SAME')
        if use_batch_normalize:
            output = tf.layers.batch_normalization(output, training=is_training)

    with tf.variable_scope("skip_connection"):
        output = output + input_layer

    output = tf.nn.relu(output)

    return output

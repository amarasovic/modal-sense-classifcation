import tensorflow as tf

class TextCNN(object):
    def __init__(self,
                 embeddings,
                 embeddings_pretrain,
                 embeddings_trainable,
                 reg_coef,
                 num_filters,
                 filter_sizes_1,
                 filter_sizes_2,
                 filter_sizes_3,
                 num_classes,
                 max_length):
        """
        CNN for modal sense classification: http://www.aclweb.org/anthology/W/W16/W16-1613.pdf
        :param embeddings: embeddings object - numpy array with shape(embeddings_number, embeddings_size)
        :param embeddings_pretrain: use pretrained embeddings or not
        :param embeddings_trainable: train embeddings or keep them fixed
        :param reg_coef: train embeddings or keep them fixed
        :param num_filters: number of filters
        :param filter_sizes_1: row size of a filter
        :param filter_sizes_2: row size of a filter
        :param filter_sizes_3: row size of a filter
        :param num_classes: number of classes
        """

        filter_sizes = [filter_sizes_1, filter_sizes_2, filter_sizes_3]
        num_filters_total = num_filters * len(filter_sizes)

        embeddings_size = embeddings.shape[1]
        embeddings_number = embeddings.shape[0]

        tf.set_random_seed(10)
        # placeholders
        self.sentences = tf.placeholder(tf.int32, shape=[None, None], name="sentences")
        self.targets = tf.placeholder(tf.int32, shape=[None, None], name="targets")
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name="dropout_keep_prob")

        weights_conv = [tf.get_variable(name="weights_conv_" + str(filter_size),
                                        shape=[filter_size, embeddings_size, 1, num_filters],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=filter_size+i*num_filters)
                                        )
                        for i, filter_size in enumerate(filter_sizes)]

        biases_conv = [tf.get_variable(name="biases_conv_" + str(filter_size),
                                       initializer=tf.constant_initializer(0.01),
                                       shape=[num_filters])
                       for filter_size in filter_sizes]

        weight_output = tf.get_variable(name="weight_output",
                                        initializer=tf.contrib.layers.xavier_initializer(seed=0),
                                        shape= [num_filters_total, num_classes])

        bias_output = tf.get_variable(name="bias_output",
                                      initializer=tf.constant_initializer(0.01),
                                      shape=[num_classes])

        if embeddings_pretrain == "True":
            embeddings_tuned = tf.get_variable("pretrained_emb",
                                                shape=[embeddings_number, embeddings_size],
                                                initializer=tf.constant_initializer(embeddings),
                                                trainable=embeddings_trainable,
                                                dtype=tf.float32)
        else:
            embeddings_tuned = tf.get_variable("random_emb",
                                                shape=[embeddings_number, embeddings_size],
                                                initializer=tf.random_uniform_initializer(-1, 1, seed=24, dtype=tf.float32),
                                                trainable=embeddings_trainable,
                                                dtype=tf.float32
                                                )

        embedded_chars = tf.nn.embedding_lookup(embeddings_tuned, self.sentences)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            # convolution layer with different filter size
            conv = tf.nn.conv2d(embedded_chars_expanded, weights_conv[i], strides=[1, 1, 1, 1], padding="VALID")
            # non-linearity
            h = tf.nn.relu(tf.nn.bias_add(conv, biases_conv[i]))
            # consider trying ELUs
            # h = tf.nn.elu(tf.nn.bias_add(conv, biases_conv[i]))

            pooled = tf.nn.max_pool(h,
                                    ksize=[1, max_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
            pooled_outputs.append(pooled)

        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        scores = tf.nn.xw_plus_b(h_drop, weight_output, bias_output)

        self.prediction = tf.nn.softmax(scores)

        tv = tf.trainable_variables()
        regularization_cost = tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, tf.cast(self.targets, tf.float32))) +\
                        reg_coef * regularization_cost
import numpy as np
import tensorflow as tf
import logging
from models.MSC_CNN import TextCNN
import time as ti
import os


def train(argv):
    embeddings = argv.embeddings

    train_sent = argv.train_sent
    train_targets = argv.train_targets
    test_sent = argv.test_sent
    test_targets = argv.test_targets

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(24)
        gpu_options = tf.GPUOptions(allow_growth=True)

        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True,
                                      gpu_options=gpu_options)

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn = TextCNN(embeddings=embeddings,
                          embeddings_pretrain=argv.pretrained_emb,
                          embeddings_trainable=argv.train_emb,
                          reg_coef=argv.reg_coef,
                          num_filters=argv.num_filters,
                          filter_sizes_1=argv.filter_sizes_1,
                          filter_sizes_2=argv.filter_sizes_2,
                          filter_sizes_3=argv.filter_sizes_3,
                          num_classes=np.asarray(train_targets, dtype=np.int32).shape[1],
                          max_length=argv.max_length)

            param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(),
                                    tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
            logging.info('Total_params: %d\n' % param_stats.total_parameters)

            global_step = tf.Variable(0, trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=argv.lr).minimize(cnn.loss)

            # output directory for models
            timestamp = str(int(ti.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir,
                                                   "runs/msc_" + argv.corpus + "_fold" + str(argv.fold),
                                                   timestamp))
            logging.info("Writing to %s " % out_dir)

            # checkpoint setup
            checkpoints_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_best = os.path.join(checkpoints_dir, "model")
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)

            saver = tf.train.Saver(tf.global_variables())

            init_vars = tf.global_variables_initializer()
            sess.run(init_vars)

            if argv.shuffle == "True":
                np.random.seed(42)
                train_data = np.asarray(list(zip(train_sent, train_targets)))
                np.random.shuffle(train_data)
                train_sent, train_targets = zip(*train_data)
                train_sent = np.asarray(train_sent)
                train_targets = np.asarray(train_targets)

            for step in range(argv.num_iter):
                offset = (step * argv.batch_size) % (train_targets.shape[0] - argv.batch_size)
                batch_sent = train_sent[offset:(offset + argv.batch_size)]
                batch_targets = train_targets[offset:(offset + argv.batch_size)]
                feed_dict = {cnn.sentences: batch_sent,
                             cnn.targets: batch_targets,
                             cnn.dropout_keep_prob: argv.drop_keep_prob}
                _, _, l, predictions = sess.run([optimizer, global_step, cnn.loss, cnn.prediction],
                                                            feed_dict)

                if not step % 100:
                    logging.info("Minibatch loss at step %s: %s " % (step,l))
                    logging.info("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_targets))

            feed_dict = {cnn.sentences: test_sent,
                         cnn.targets: test_targets,
                         cnn.dropout_keep_prob: 1.0}
            test_predictions = sess.run([cnn.prediction], feed_dict=feed_dict)
            test_predictions = np.asarray(test_predictions).reshape(test_targets.shape)
            test_accuracy = accuracy(test_predictions, np.asarray(test_targets))

            path = saver.save(sess, checkpoint_best)
            logging.info("Saved the model checkpoint to {}\n".format(path))

            return test_accuracy


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

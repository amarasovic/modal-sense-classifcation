import tensorflow as tf
import numpy as np
import logging


def eval(argv):
    test_sent = argv.test_sent
    test_targets = argv.test_targets

    checkpoint_file = tf.train.latest_checkpoint(argv.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            sentences_pl = graph.get_operation_by_name("sentences").outputs[0]
            targets_pl = graph.get_operation_by_name("targets").outputs[0]
            dropout_keep_prob_pl = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            predictions_op = graph.get_operation_by_name("predictions").outputs[0]

            feed_dict = {sentences_pl: test_sent,
                         targets_pl: test_targets,
                         dropout_keep_prob_pl: 1.0}
            test_predictions = sess.run([predictions_op], feed_dict=feed_dict)
            test_predictions = np.asarray(test_predictions).reshape(test_targets.shape)
            test_accuracy = accuracy(test_predictions, np.asarray(test_targets))
            logging.info('test accuracy: ' + str(test_accuracy))


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


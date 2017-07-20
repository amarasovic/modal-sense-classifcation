import argparse
from get_data import get_emb, get_data, get_masc_data, get_vocabulary
from train import train
from eval import eval
import logging
import numpy as np
import json
if __name__ == '__main__':
    # parameters
    parser = argparse.ArgumentParser(description='modal sense classification', add_help=False,
                                     conflict_handler='resolve')

    parser.add_argument('-mode', default='train', help='train/test')

    """ architecture """
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embeddings')
    parser.add_argument('--emb_type', default="w2v", help="type of embeddings")
    parser.add_argument('--num_filters', type=int, default=100, help="number of filters per size")
    parser.add_argument('--filter_sizes_1', type=int, default=3, help="the first choice for the size of filters")
    parser.add_argument('--filter_sizes_2', type=int, default=4, help="the second choice for the size of filters")
    parser.add_argument('--filter_sizes_3', type=int, default=5, help="the third choice for the size of filters")
    parser.add_argument('--drop_keep_prob', type=float, default=0.5, help='drop keep prob')

    """ training options """
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--reg_coef', type=float, default=1e-3, help='L2 Reg rate')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size')
    parser.add_argument('--num_iter', type=int, default=1001, help='number of iterations to train')
    parser.add_argument("--train_emb", type=str, default="False", help="train embeddings or not")
    parser.add_argument("--pretrained_emb", type=str, default="True", help="pretrained emb or not")
    parser.add_argument('--corpus', default=None, help='corpus for training')
    parser.add_argument('--shuffle', default="False", help='shuffle training data')
    parser.add_argument('--drop_mv', default="False", help='drop mv from sent')
    parser.add_argument('--test_corpus', default=None, help='test corpus for training')

    argv = parser.parse_args()

    logging.basicConfig(filename='logs/' + argv.corpus + '.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # construct vocabulary from train and test data
    vocabulary, max_length = get_vocabulary(argv.corpus)
    parser.add_argument('--max_length', type=int, default=max_length, help='max sentence length')


    logging.info("Saving vocabulary for testing...")
    vocab_dict = {'vocabulary': vocabulary}
    with open("vocabs/" + argv.corpus + '_vocab.json', 'w') as fp:
        json.dump({'vocabulary':vocabulary}, fp)

    # get pre-trained embeddings
    logging.info("Get embeddings...")
    embeddings, _ = get_emb(argv.emb_type, vocabulary, argv.emb_size)

    # modal_verbs = ['koennen', 'muessen', 'sollen', 'duerfen']
    modal_verbs = ["can", "could", "may", "must", "should"]
    if argv.drop_mv == "True":
        for modal in modal_verbs:
            index = vocabulary[modal]
            embeddings[index, :] = np.zeros((1, argv.emb_size))

    parser.add_argument('--embeddings', default=embeddings, help="embeddings matrix")

    # get data
    data = get_data(argv.corpus, max_length, vocabulary)

    train_sentences = data['train_sentences']
    train_labels = data['train_labels']
    test_sentences = data['test_sentences']
    test_labels = data['test_labels']

    accuracies = []
    for k in range(5):
        train_sent = train_sentences[k]
        train_targets = train_labels[k]
        test_sent = test_sentences[k]
        test_targets = test_labels[k]

        parser.add_argument('--train_sent', default=train_sent, help='train sent')
        parser.add_argument('--train_targets', default=train_targets, help='train targets')
        parser.add_argument('--test_sent', default=test_sent, help='test sent')
        parser.add_argument('--test_targets', default=test_targets, help='test targets')
        parser.add_argument('--fold', type=int, default=k+1, help='current fold')

        argv = parser.parse_args()

        accuracy_fold = train(argv)
        logging.info('fold ' + str(k+1) + ": " + str(accuracy_fold))
        accuracies.append(accuracy_fold)

    accuracy = sum(accuracies) / float(len(accuracies))
    logging.info('macro average accuracy: ' + str(accuracy))

    if argv.mode == "eval":
        # load MASC data
        genres = ['blog', 'court-transcript', 'debate-transcript', 'email', 'essays', 'face-to-face', 'ficlets',
                  'fiction', 'govt-docs', 'jokes', 'journal', 'letters', 'movie-script', 'newspaper', 'non-fiction',
                  'technical', 'telephone', 'travel-guides', 'twitter']
        # use trained models

        json_vocab = "vocabs/" + argv.corpus + '_vocab.json'
        with open(json_vocab) as data_file:
            vocab = json.load(data_file)
        vocabulary = vocab['vocabulary']

        data = get_masc_data(argv.test_corpus, vocabulary)

        parser.add_argument('--test_sent', default=test_sent, help='test sent')
        parser.add_argument('--test_targets', default=test_targets, help='test targets')
        parser.add_argument('--checkpoint_dir',
                            default="runs/msc_" + argv.corpus + "_fold1",
                            help="checkpoint directory from training run")

        argv = parser.parse_args()

        eval(argv)


















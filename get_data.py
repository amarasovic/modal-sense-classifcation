from gensim import models
import numpy as np
import re
import logging
from nltk.tokenize import word_tokenize
import codecs
from collections import Counter
import itertools


def clean_str(string):
    """
    Tokenization/string cleaning
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_vocabulary(corpus):
    """
    Build vocabulary from train and test data
    :param corpus: mv_(un)balance_classifier#
    :return: vocabulary
    """
    sentences_all_folds = []
    max_length = -1

    for k in range(1, 6):
        path_train = "data/EPOS_E/train/" + corpus + "_fold" + str(k)+".txt"
        #path_train = "data/EPOS_G/train/"+ argv.corpus + ".txt"
        train_sentences = codecs.open(path_train, "r").readlines()
        train_sentences_token = [word_tokenize(clean_str(s))for s in train_sentences]

        path_test = "data/EPOS_E/test/" + corpus + "_fold" + str(k)+ ".txt"
        #path_test = "data/EPOS_G/test/"+ argv.corpus + ".txt"
        test_sentences = codecs.open(path_test, "r").readlines()
        test_sentences_token = [word_tokenize(clean_str(s)) for s in test_sentences]

        sentences = train_sentences_token + test_sentences_token
        sentences_all_folds.extend(sentences)
        max_length = max(max_length, max(len(s) for s in sentences))

    logging.info("Building vocabulary...")
    word_counts = dict(Counter(itertools.chain(*sentences_all_folds)).most_common())
    word_counts_list = zip(word_counts.keys(), word_counts.values())

    PAD = "<PAD>"
    UNK = "<UNK>"
    vocabulary_inv = [x[0] for x in word_counts_list]
    vocabulary_inv.append(PAD)
    vocabulary_inv.append(UNK)

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return vocabulary, max_length


def word2vec_emb_vocab(vocabulary, dim, emb_type):
    """
    :param vocabulary: vocabulary constructed from the train & test
    :param dim: dimension of the embeddings
    :param emb_type: glove or w2v
    :return: numpy array w/ shape [size of the vocab, dim]
    """
    PAD = "<PAD>"
    UNK = "<UNK>"

    if emb_type == "w2v":
        logging.info("Loading pre-trained w2v binary file...")
        w2v_model = models.Word2Vec.load_word2vec_format('../embeddings/GoogleNews-vectors-negative300.bin', binary=True)

    else:
        # convert glove vecs into w2v format: https://github.com/manasRK/glove-gensim/blob/master/glove-gensim.py
        glove_file = "../embeddings/glove/glove_"+str(dim)+"_w2vformat.txt"
        w2v_model = models.Word2Vec.load_word2vec_format(glove_file, binary=False)  # GloVe Model

    emb_w2v = w2v_model.wv.syn0

    logging.info("Building embeddings for this dataset...")
    vocab_size = len(vocabulary)
    embeddings = np.zeros((vocab_size, dim), dtype=np.float32)

    embeddings[vocabulary[PAD],:] = np.zeros((1, dim))
    embeddings[vocabulary[UNK],:] = np.mean(emb_w2v, axis=0).reshape((1, dim))

    counter = 0
    for word in vocabulary:
        try:
            embeddings[vocabulary[word], :] = w2v_model[word].reshape((1, dim))
        except KeyError:
            counter += 1
            embeddings[vocabulary[word], :] = embeddings[vocabulary[UNK],:]

    logging.info("Number of out-of-vocab words: %s from %s" % (counter, vocab_size))

    del emb_w2v
    del w2v_model

    assert len(vocabulary) == embeddings.shape[0]

    return embeddings, vocabulary


def get_emb(emb_type, vocabulary, dim):
    if emb_type == "w2v":
        emb, vocab = word2vec_emb_vocab(vocabulary, dim, emb_type)

    if emb_type == "glove":
        emb, vocab = word2vec_emb_vocab(vocabulary, dim, emb_type)
    return emb, vocab


def get_data(corpus, max_length, vocabulary):
    """
    Get list of train/test datasets of every fold
    :param corpus: mv_(un)balance_classifier#
    :param max_length: maximum sentence length
    :param vocabulary: vocabulary
    :return: dictionary of lists of train/test sentences/labels of every fold
    """
    train_sentences = []
    train_labels = []
    test_sentences = []
    test_labels = []

    for k in range(1,6):
        path_train = "data/EPOS_E/train/" + corpus + "_fold" + str(k) + ".txt"
        #path_train = "data/EPOS_G/train/"+ corpus + ".txt"
        sentences_train, labels_train = get_data_fold(path_train)

        path_test = "data/EPOS_E/test/" + corpus + "_fold" + str(k) + ".txt"
        # path_test = "data/EPOS_G/test/"+ corpus + ".txt"
        sentences_test, labels_test = get_data_fold(path_test)

        train_dataset, train_label = build_input_data(pad_sentences(sentences_train, max_length),
                                                      labels_train,
                                                      vocabulary)

        test_dataset, test_label = build_input_data(pad_sentences(sentences_test, max_length),
                                                    labels_test,
                                                    vocabulary)
        train_sentences.append(train_dataset)
        train_labels.append(train_label)
        test_sentences.append(test_dataset)
        test_labels.append(test_label)

    data = {'train_sentences': train_sentences,
            'train_labels': train_labels,
            'test_sentences': test_sentences,
            'test_labels': test_labels
            }
    return data


def get_data_fold(sentences_file):
    """
    :param sentences_file: file w/ the format: SENTENCE TAB LABEL
    :return: list of tokenized sentences and list of labels encoded as one-hot vectors
    """
    lines = codecs.open(sentences_file, "r").readlines()

    sentences_list = []
    labels_list = []

    for line in lines:
        line_split = line.split("\t")
        sentences_list.append(line_split[0])
        labels_list.append(line_split[1].split("\n")[0])

    sentences = [word_tokenize(clean_str(s)) for s in sentences_list]

    labels_set = {'ep': 0, 'de': 1, 'dy': 2}
    num_of_classes = len(labels_set)

    labels = []
    for label in labels_list:
        temp = [0]*num_of_classes
        index = labels_set[label]
        temp[index] = 1
        labels.append(temp)

    labels = np.asarray(labels)
    return [sentences, labels]


def get_masc_data(corpus, vocabulary):
    path_test = "..corpora/MSC_data/MASC/" + corpus + ".txt"
    sentences_test, labels_test = get_data_fold(path_test)

    max_length = max(len(s) for s in sentences_test)

    test_dataset, test_label = build_input_data(pad_sentences(sentences_test, max_length),
                                                    labels_test,
                                                    vocabulary)

    data = {'test_sentences': test_dataset,
            'test_labels': test_label
            }

    return data


def pad_sentences(sentences, sentence_length, padding_word="<PAD>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sentence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])

    y = np.array(labels)
    return [x, y]



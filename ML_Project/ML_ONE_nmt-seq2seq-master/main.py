import os
import code
import numpy as np
import pickle
import nltk
from collections import Counter
import config
import utils


def main(args):
    train_en, train_cn = utils.load_data(args.train_file)
    dev_en, dev_cn = utils.load_data(args.dev_file)
    num_train = len(train_en)
    num_dev = len(dev_en)

    en_dict, en_total_words = utils.build_dict(train_en)
    cn_dict, cn_total_words = utils.build_dict(train_cn)
    args.en_total_words = en_total_words
    args.cn_total_words = cn_total_words

    inv_en_dict = {v:k for k, v in en_dict.items()}
    inv_cn_dict = {v:k for k, v in cn_dict.items()}

    train_en, train_cn = utils.encode(train_en, train_cn, en_dict, cn_dict)
    dev_en, dev_cn = utils.encode(dev_en, dev_cn,en_dict, cn_dict)

    train_data = utils.gen_examples(train_en, train_cn, args.batch_size)
    dev_data = utils.gen_examples(dev_en, dev_cn, args.batch_size)
    print(len(train_data[6][0]))

if __name__ == "__main__":
    args = config.get_args()
    main(args)
    # train_file = 'data/train_mini.txt'
    # dev_file = 'data/dev_mini.txt'
    # main(train_file, dev_file)




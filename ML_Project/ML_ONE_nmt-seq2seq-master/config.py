import argparse
def get_args():
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--train_file', type=str, default='data/train_mini.txt',
                        help="training file")
    parser.add_argument('--dev_file', type=str, default='data/dev_mini.txt',
                        help="development file")
    #model
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size for training")

    return parser.parse_args()
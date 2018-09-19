import nltk
import io
from collections import Counter
import numpy as np

def load_data(in_file):
    en = []
    cn = []
    with open(in_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            en.append(['BOS']+nltk.word_tokenize(line[0])+['EOS'])
            cn.append(['BOS']+[i for i in line[1]]+['EOS'])
    return en,cn

def build_dict(sentences, max_words = 50000):
    word_coun = Counter()
    for sentence in sentences:
        for s in sentence:
            word_coun[s]+=1
    # 得到出现频率最高的单词集合
    ls = word_coun.most_common(max_words)
    #长度加1，设置空值
    total_words = len(ls)+1
    #enumerate会返回下标从0开始，所以返回上面的高频单词，用索引来代替它们
    word_dict = {w[0]: index+1 for (index, w) in enumerate(ls)}
    word_dict['NUK'] = 0
    return word_dict, total_words
#将句子变成数字指向
def encode(en_sentences, cn_sentences, en_dict, cn_dict):
    length = len(en_sentences)
    out_en_sentences = []
    out_cn_sentences = []
    for i in range(length):
        en_seq = [en_dict[w] if w in en_dict else 0 for w in en_sentences[i]]
        cn_seq = [cn_dict[w] if w in cn_dict else 0 for w in cn_sentences[i]]
        out_en_sentences.append(en_seq)
        out_cn_sentences.append(cn_seq)
    #将句子按照句子的长短进行排序
    #range(len(seq))其实就是索引，lambda隐函数按照句子的len进行排序
    def len_argsort(seq):
        return sorted(range(len(seq)), key = lambda x:len(seq[x]))
    sorted_index = len_argsort(out_en_sentences)
    out_en_sentences = [out_en_sentences[i] for i in sorted_index]
    out_cn_sentences = [out_cn_sentences[i] for i in sorted_index]
    return out_en_sentences, out_cn_sentences
#构建数据矩阵，大小为batch_size *seq_length
#将数据打包成一份一份的矩阵，分别输入，前面的排序就是为这里做准备
#主要的目的是为了加开运算速度
def gen_examples(en_sentences, cn_sentences, batch_size):
    minibatches = get_minibatches(len(en_sentences), batch_size)
    all_ex = []
    for minibatch in minibatches:
        mb_en_sentences = [en_sentences[t] for t in minibatch]
        #将数据变成numpy array
        mb_x, mb_x_mask = prepare_data(mb_en_sentences)
        #中英文每条数据对应长度相同
        mb_cn_sentences = [cn_sentences[t] for t in minibatch]
        mb_y, mb_y_mask = prepare_data(mb_cn_sentences)
        all_ex.append((mb_x, mb_x_mask, mb_y, mb_y_mask))

    return all_ex


#将排序好的数据按batch_size进行等分，不明白为啥要shuffle（默认没有）
#并返回没个每个矩阵行对应的索引值（每行代表一条数据）
def get_minibatches(n, minibatches_size, shuffle = False):
    minibatches = []
    idx_list = np.arange(0, n, minibatches_size)
    if shuffle:
        #shuffle：将数据打乱，像洗牌一样
      np.random.shuffle(idx_list)
    for idx in idx_list:
        #此处加上min()是为了决定当时最后一份数据长度不满足minibatches_size是，直接取到末尾
        minibatches.append(np.arange(idx, min(n, idx + minibatches_size)))
    return minibatches

#构造没份数据为numpy array
def prepare_data(seqs):
    B = len(seqs) #B = 128
    lengths = [len(seq) for seq in seqs]
    max_len = np.max(lengths)
    x = np.zeros((B, max_len)).astype('int32')
    x_mask = np.zeros((B, max_len)).astype('float32')
    #填充矩阵，因为即使两条数据分到同一个矩阵也不能肯定两者同等长
    #所以该份数据长度最大的作为维度，下面是填充矩阵
    for idx, seq in enumerate(seqs):
        #感觉这里很巧妙但有感觉很普通，思路满分
        x[idx, :lengths[idx]] = seq
        #这里表示原数据的长度所及的位置都填充，为1，补齐的为0
        #这里可以看出，相当于有两个矩阵
        x_mask[idx, :lengths[idx]] = 1.
    return x, x_mask

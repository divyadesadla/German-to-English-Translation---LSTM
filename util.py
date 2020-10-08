import torch
import math
from torch.nn.init import kaiming_uniform
from core import LSTM_cell, StackedLSTM, Attention, OutputLayer, Seq2SeqModel, Seq2SeqAttentionModel
import pickle
from typing import Dict, List, Tuple


def input_transpose(sents, pad_token):
    """
    This function transforms a list of sentences of shape (batch_size, token_num) into 
    a list of shape (token_num, batch_size). You may find this function useful if you
    use pytorch
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

class Vocab(object):
    def __init__(self, src_sents, tgt_sents, vocab_size, freq_cutoff):
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        self.src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        self.tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

    def __repr__(self):
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))

    
class VocabEntry(object):
    def __init__(self):
        self.word2id = dict()
        self.unk_id = 3
        self.word2id['<pad>'] = 0
        self.word2id['<s>'] = 1
        self.word2id['</s>'] = 2
        self.word2id['<unk>'] = 3

        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        return word in self.word2id

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def add(self, word):
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        vocab_entry = VocabEntry()

        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(f'number of word types: {len(word_freq)}, number of word types w/ frequency >= {freq_cutoff}: {len(valid_words)}')

        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)

        return vocab_entry

    
def load_data() -> (Tuple[List[int], List[int]], Tuple[List[int], List[int]], Dict[int, str], Dict[int, str]):
    """ Load the dataset.
    :return: (1) train_sentences: list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) test_sentences : list of (source_sentence, target_sentence) pairs, where both source_sentence
                                  and target_sentence are lists of ints
             (2) source_vocab   : dictionary which maps from source word index to source word
             (3) target_vocab   : dictionary which maps from target word index to target word
    """
    with open('data/corpus.bin', 'rb') as f:
        corpus = pickle.load(f)
        test_sentences = corpus[:1000]
        train_sentences = corpus[1000:]
        print("# train sentences: {}\n"
              "# test sentences: {}\n".format(len(train_sentences), len(test_sentences)))
        
    with open('data/vocab.bin', 'rb') as f:
        vocab = pickle.load(f)
    print(vocab)
    
    return train_sentences, test_sentences, vocab


def uniform_tensor(shape, a, b):
    return torch.FloatTensor(*shape).uniform_(a, b)


def normal_tensor(shape):
    return torch.FloatTensor(*shape).normal_()


def kaiming_tensor(shape):
    tensor = torch.FloatTensor(*shape)
    torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))
    return tensor


def initialize_bias(weight, bias):
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    torch.nn.init.uniform_(bias, -bound, bound)


# hyperparameters

SOURCE_VOCAB_SIZE = 2896
TARGET_VOCAB_SIZE = 2931
SOURCE_EMBEDDING_DIM = 400
TARGET_EMBEDDING_DIM = 400
HIDDEN_DIM = 500


def initialize_seq2seq_params() -> Dict[str, torch.tensor]:
    """ Random initialization of weights for a Seq2Seq model.
    :return: model_params, a dictionary Dict[str, torch.tensor] mapping from parameter name to parameter value
    """
    stdv = 1.0 / math.sqrt(HIDDEN_DIM)
    model_params = {
        'enc_1_Wxi': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_1_Wxf': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_1_Wxc': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_1_Wxo': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        

        'enc_2_Wxi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_2_Wxf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_2_Wxc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_2_Wxo': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        

        'dec_1_Wxi': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_1_Wxf': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_1_Wxc': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_1_Wxo': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),        
        
        
        'dec_2_Wxi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_2_Wxf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_2_Wxc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_2_Wxo': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        "source_embedding_matrix": normal_tensor((SOURCE_VOCAB_SIZE, SOURCE_EMBEDDING_DIM)),
        "target_embedding_matrix": normal_tensor((TARGET_VOCAB_SIZE, TARGET_EMBEDDING_DIM)),

        "output_weight": kaiming_tensor((TARGET_VOCAB_SIZE, HIDDEN_DIM)),
        "output_bias": torch.FloatTensor(TARGET_VOCAB_SIZE),
    }

    initialize_bias(model_params["output_weight"], model_params["output_bias"])
    return model_params


def build_seq2seq_model(model_params: Dict[str, torch.tensor]) -> Seq2SeqModel:
    """ Build a Seq2SeqModel object from a model_params dict """
    # Wxi, Whi, Wxf, Whf, Wxc, Whc, Wxo, Who
    for key in model_params:
        model_params[key].requires_grad_(True)
    
    encoder_lstm = StackedLSTM([
        LSTM_cell(model_params['enc_1_Wxi'], 
                  model_params['enc_1_Whi'],
                  model_params['enc_1_Wxf'],
                  model_params['enc_1_Whf'],
                  model_params['enc_1_Wxc'],
                  model_params['enc_1_Whc'],
                  model_params['enc_1_Wxo'],
                  model_params['enc_1_Who']),
        LSTM_cell(model_params['enc_2_Wxi'], 
                  model_params['enc_2_Whi'],
                  model_params['enc_2_Wxf'],
                  model_params['enc_2_Whf'],
                  model_params['enc_2_Wxc'],
                  model_params['enc_2_Whc'],
                  model_params['enc_2_Wxo'],
                  model_params['enc_2_Who']),
    ])
    decoder_lstm = StackedLSTM([
        LSTM_cell(model_params['dec_1_Wxi'], 
                  model_params['dec_1_Whi'],
                  model_params['dec_1_Wxf'],
                  model_params['dec_1_Whf'],
                  model_params['dec_1_Wxc'],
                  model_params['dec_1_Whc'],
                  model_params['dec_1_Wxo'],
                  model_params['dec_1_Who']),
        LSTM_cell(model_params['dec_2_Wxi'], 
                  model_params['dec_2_Whi'],
                  model_params['dec_2_Wxf'],
                  model_params['dec_2_Whf'],
                  model_params['dec_2_Wxc'],
                  model_params['dec_2_Whc'],
                  model_params['dec_2_Wxo'],
                  model_params['dec_2_Who']),
    ])
    source_embedding_matrix = model_params['source_embedding_matrix']
    target_embedding_matrix = model_params['target_embedding_matrix']
    output_layer = OutputLayer(model_params['output_weight'], model_params['output_bias'])
    return Seq2SeqModel(HIDDEN_DIM, encoder_lstm, decoder_lstm, source_embedding_matrix,
                        target_embedding_matrix, output_layer)

def initialize_seq2seq_attention_params() -> Dict[str, torch.tensor]:
    """ Random initialization of weights for a Seq2SeqAttention model.
    :return: model_params, a dictionary Dict[str, torch.tensor] mapping from parameter name to parameter value"""
    stdv = 1.0 / math.sqrt(HIDDEN_DIM)
    model_params = {
        'enc_1_Wxi': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_1_Wxf': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_1_Wxc': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_1_Wxo': uniform_tensor((HIDDEN_DIM, SOURCE_EMBEDDING_DIM), -stdv, stdv),
        'enc_1_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        
        'enc_2_Wxi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_2_Wxf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'enc_2_Wxc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'enc_2_Wxo': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'enc_2_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        
        'dec_1_Wxi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_1_Wxf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_1_Wxc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_1_Wxo': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM + TARGET_EMBEDDING_DIM), -stdv, stdv),
        'dec_1_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        
        'dec_2_Wxi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whi': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_2_Wxf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whf': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),

        'dec_2_Wxc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Whc': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        'dec_2_Wxo': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        'dec_2_Who': uniform_tensor((HIDDEN_DIM, HIDDEN_DIM), -stdv, stdv),
        
        "source_embedding_matrix": normal_tensor((SOURCE_VOCAB_SIZE, SOURCE_EMBEDDING_DIM)),
        "target_embedding_matrix": normal_tensor((TARGET_VOCAB_SIZE, TARGET_EMBEDDING_DIM)),

        "output_weight": kaiming_tensor((TARGET_VOCAB_SIZE, 2* HIDDEN_DIM)),
        "output_bias": torch.FloatTensor(TARGET_VOCAB_SIZE),

        "attention": torch.eye(HIDDEN_DIM)
    }

    initialize_bias(model_params["output_weight"], model_params["output_bias"])
    return model_params


def build_seq2seq_attention_model(model_params: Dict[str, torch.tensor]) -> Seq2SeqAttentionModel:
    """ Build a Seq2SeqAttentionModel object from a model_params dict """

    for key in model_params:
        model_params[key].requires_grad_(True)

    encoder_lstm = StackedLSTM([
        LSTM_cell(model_params['enc_1_Wxi'], 
                  model_params['enc_1_Whi'],
                  model_params['enc_1_Wxf'],
                  model_params['enc_1_Whf'],
                  model_params['enc_1_Wxc'],
                  model_params['enc_1_Whc'],
                  model_params['enc_1_Wxo'],
                  model_params['enc_1_Who']),
        LSTM_cell(model_params['enc_2_Wxi'], 
                  model_params['enc_2_Whi'],
                  model_params['enc_2_Wxf'],
                  model_params['enc_2_Whf'],
                  model_params['enc_2_Wxc'],
                  model_params['enc_2_Whc'],
                  model_params['enc_2_Wxo'],
                  model_params['enc_2_Who']),
    ])
    decoder_lstm = StackedLSTM([
        LSTM_cell(model_params['dec_1_Wxi'], 
                  model_params['dec_1_Whi'],
                  model_params['dec_1_Wxf'],
                  model_params['dec_1_Whf'],
                  model_params['dec_1_Wxc'],
                  model_params['dec_1_Whc'],
                  model_params['dec_1_Wxo'],
                  model_params['dec_1_Who']),
        LSTM_cell(model_params['dec_2_Wxi'], 
                  model_params['dec_2_Whi'],
                  model_params['dec_2_Wxf'],
                  model_params['dec_2_Whf'],
                  model_params['dec_2_Wxc'],
                  model_params['dec_2_Whc'],
                  model_params['dec_2_Wxo'],
                  model_params['dec_2_Who']),
    ])

    source_embedding_matrix = model_params['source_embedding_matrix']
    target_embedding_matrix = model_params['target_embedding_matrix']
    attention = Attention(model_params['attention'])
    output_layer = OutputLayer(model_params['output_weight'], model_params['output_bias'])
    return Seq2SeqAttentionModel(HIDDEN_DIM, encoder_lstm, decoder_lstm, source_embedding_matrix,
                                 target_embedding_matrix, attention, output_layer)

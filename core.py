""" This file contains components needed by both with-attention and without-attention variants. """
import torch
from typing import List, Union
from torch.nn.functional import softmax

SOS_token = 0
EOS_token = 1

class LSTM_cell(object):

    def __init__(self, Wxi, Whi, Wxf, Whf, Wxc, Whc, Wxo, Who):
        """
        :param Wxi, Wxf, Wxc, Wxo: a tensor of shape [hidden_dim, source_embedding_dim] for the first layer and [hidden_dim, hidden_dim] for the second layer.
        :param Whi, Whf, Whc, Who: a tensor of shape [hidden_dim, hidden_dim].
        Details of dimensionality can be checked in the functions initialize_seq2seq_params and initialize_seq2seq_attention_params in util.py
        """
        self.Wxi = Wxi
        self.Whi = Whi
        self.Wxf = Wxf
        self.Whf = Whf
        self.Wxc = Wxc
        self.Whc = Whc
        self.Wxo = Wxo
        self.Who = Who

        

    def forward(self, input, last_hidden, last_cell):
        self.input = input
        self.last_hidden = last_hidden
        self.last_cell = last_cell

        forget_gate = torch.sigmoid(torch.mm(self.Wxf,self.input) + torch.mm(self.Whf,self.last_hidden))
        input_gate = torch.sigmoid(torch.mm(self.Wxi,self.input) + torch.mm(self.Whi,self.last_hidden))
        candidate_cell_state = torch.tanh(torch.mm(self.Wxc,self.input) + torch.mm(self.Whc,self.last_hidden))
        current_cell_state = forget_gate * self.last_cell + input_gate * candidate_cell_state
        out_sigmoid = torch.sigmoid(torch.mm(self.Wxo,self.input) + torch.mm(self.Who,self.last_hidden))
        out_hidden_state = out_sigmoid * torch.tanh(current_cell_state)

        current_cell_state = torch.flatten(current_cell_state)
        out_hidden_state = torch.flatten(out_hidden_state)
        
        return out_hidden_state,current_cell_state

        """
        :param input: a tensor of shape [input_dim], denoted x_j in the assignment
        :param last_hidden: a tensor of shape [L, hidden_dim], denoted H_{j-1} in the assignment
        :param last_cell: a tensor of shape [L, hidden_dim], denoted C_{j-1} in the assignment
        :return: hidden, a tensor of shape [L, hidden_dim], denoted H_j in the assignment
        :return: cell, a tensor of shape [L, hidden_dim], denoted C_j in the assignment
        """
        raise NotImplementedError
        

class StackedLSTM(object):
    """ A multilevel stack of LSTMs.
    """

    def __init__(self, lstms: List[LSTM_cell]):
        """
        :param lstms: the LSTMs in the stack, from bottom to top.  The LSTMs should share the same
                     hidden unit dimension, `hidden_dim`.  The bottom-most LSTM should have an input dimension
                      of `input_dim`; all other LSTMs should have an input dimension of `hidden_dim`.
        """
        self.lstms = lstms

    def forward(self, input: torch.tensor, last_hidden: torch.tensor, last_cell: torch.tensor):
        self.input = input
        self.last_hidden = last_hidden
        self.last_cell = last_cell

        stack1_down_hidden, stack1_down_current = LSTM_cell.forward(self.lstms[0],self.input[None].t(),self.last_hidden[0][None].t(),self.last_cell[0][None].t())
        stack1_up_hidden, stack1_up_current = LSTM_cell.forward(self.lstms[1],stack1_down_hidden[None].t(),self.last_hidden[1][None].t(),self.last_cell[1][None].t())

        hidden = torch.stack([stack1_down_hidden,stack1_up_hidden],dim=0)
        current_cell = torch.stack([stack1_down_current,stack1_up_current],dim=0)

        return hidden, current_cell

        """
        :param input: a tensor of shape [input_dim], denoted x_j in the assignment
        :param last_hidden: a tensor of shape [L, hidden_dim], denoted H_{j-1} in the assignment
        :param last_cell: a tensor of shape [L, hidden_dim], denoted C_{j-1} in the assignment
        :return: hidden, a tensor of shape [L, hidden_dim], denoted H_j in the assignment
        """
        raise NotImplementedError


class OutputLayer(object):
    """ A fully-connected layer that returns softmax(W^{out} h + b^{out}), where h is the input.
    """

    def __init__(self, weight: torch.tensor, bias: torch.tensor):
        """
        :param weight: [WITHOUT ATTENTION] tensor of shape [target_vocab_size, hidden_dim],
                       [WITH ATTENTION]    tensor of shape [target_vocab_size, 2 * hidden_dim],
                         denoted W^{out} in the assignment
        :param bias: tensor of shape [target_vocab_size], denoted b^{out} in the assigment
        """
        self.weight = weight
        self.bias = bias

    def forward(self, input: torch.tensor) -> torch.tensor:
        self.input = input
        probs = torch.softmax(torch.mm(self.weight,self.input) + self.bias[None].t(),dim=0)
        probs = torch.flatten(probs)
        return probs

        ## Might need to change later and put if-else condition for attention and w/o attention
        """
        :param input: [WITHOUT ATTENTION] a tensor of shape [hidden_size]
                      [WITH ATTENTION]    a tensor of shape [2*hidden_size]
        :return: probs: a tensor of shape [target_vocab_size]
        """
        raise NotImplementedError


class Attention(object):

    def __init__(self, A: torch.tensor):
        """
        :param A: a tensor of shape [hidden_dim, hidden_dim]
        """
        self.A = A

    def forward(self, source_top_hiddens: torch.tensor, target_top_hidden: torch.tensor):

        self.source_top_hiddens = source_top_hiddens
        self.target_top_hidden = target_top_hidden

        score_1 = torch.mm(self.source_top_hiddens,self.A)
        score = torch.mm(score_1,self.target_top_hidden)
        alpha = torch.softmax(score,dim=0)
        alpha = torch.flatten(alpha)
        return alpha

    
        """
        :param source_top_hiddens: tensor of shape [source_sentence_length, hidden_dim] - the hidden states
                                   from the TOP LAYER of the encoder
        :param target_top_hidden: tensor of shape [hidden_dim] - the current hidden state in the TOP LAYER of the decoder
        :return: a tensor of attention weights of shape [source_sentence_length]

        """
        raise NotImplementedError


class Seq2SeqModel(object):

    def __init__(self, hidden_dim: int, encoder: StackedLSTM, decoder: StackedLSTM,
                 source_embedding_matrix: torch.tensor, target_embedding_matrix: torch.tensor,
                 output_layer: OutputLayer):
        """
        :param encoder: the encoder StackedLSTM, with input dim `source_embedding_dim`, and hidden dim `hidden_dim`
        :param decoder: the decoder StackedLSTM, with input dim `target_embedding_dim` and hidden dim `hidden_dim`
        :param source_embedding_matrix: a tensor of shape [source_vocab_size, source_embedding_dim]
        :param target_embedding_matrix: a tensor of shape [target_vocab_size, target_embedding_dim]
        :param output_layer: an OutputLayer with input dimension `hidden_dim` and output dimension `target_vocab_size`
        """
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder_lstm = decoder
        self.source_embedding_matrix = source_embedding_matrix
        self.target_embedding_matrix = target_embedding_matrix
        self.output_layer = output_layer


class Seq2SeqAttentionModel(object):

    def __init__(self, hidden_dim: int, encoder: StackedLSTM, decoder: StackedLSTM,
                 source_embedding_matrix: torch.tensor, target_embedding_matrix: torch.tensor,
                 attention: Attention, output_layer: OutputLayer):
        """
        :param encoder: the encoder StackedLSTM, with input dim `source_embedding_dim`, and hidden dim `hidden_dim`
        :param decoder: the decoder StackedLSTM, with input dim `target_embedding_dim + hidden_dim`
                        and hidden dim `hidden_dim`
        :param source_embedding_matrix: a tensor of shape [source_vocab_size, source_embedding_dim]
        :param target_embedding_matrix: a tensor of shape [target_vocab_size, target_embedding_dim]
        :param attention: an Attention
        :param output_layer: an OutputLayer with input dimension `2*hidden_dim` and output dimension `target_vocab_size`
        """
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.decoder_lstm = decoder
        self.source_embedding_matrix = source_embedding_matrix
        self.target_embedding_matrix = target_embedding_matrix
        self.attention = attention
        self.output_layer = output_layer


def encode_all(source_sentence: List[int], model: Union[Seq2SeqModel, Seq2SeqAttentionModel]) -> torch.tensor:
    
    hidden_out = []
    cell_out = []

    source_hiddens = torch.zeros([2,model.hidden_dim])
    source_cells = torch.zeros([2,model.hidden_dim])

    for i in source_sentence:
        source_hiddens, source_cells = model.encoder.forward(model.source_embedding_matrix[i],source_hiddens,source_cells)
        hidden_out.append(source_hiddens)
        cell_out.append(source_cells)
    source_hidden = torch.stack(hidden_out, dim=0)
    source_cell = torch.stack(cell_out, dim=0)
    return source_hidden, source_cell


    """ Encode the whole source sentence.

    :param source_sentence: list of ints of length `source_sentence_length`
    :param model: either a Seq2SeqModel or a Seq2SeqAttentionModel
    :return: tensor `source_hiddens` of shape [source_sentence_length, L, hidden_dim], denoted H^{enc}_1 ... H^{enc}_S
             in the assignment
    """
    raise NotImplementedError

import argparse
import time
import torch
from typing import *
from util import *
from time import time
from torch import optim
from core import Seq2SeqAttentionModel, encode_all
from math import exp, log
from core import SOS_token, EOS_token


def decode(prev_hidden: torch.tensor, prev_cell: torch.tensor, source_hiddens: torch.tensor, prev_context: torch.tensor,
           input: int, model: Seq2SeqAttentionModel) -> (
        torch.tensor, torch.tensor,torch.tensor, torch.tensor, torch.tensor, torch.tensor):
 

    target_hiddens = torch.zeros([2,model.hidden_dim])
    target_cells = torch.zeros([2,model.hidden_dim]) 

    target_hiddens, target_cells = model.decoder_lstm.forward(torch.cat((model.target_embedding_matrix[input],prev_context),dim=0),prev_hidden,prev_cell)

    alpha = model.attention.forward(source_hiddens[:,-1],target_hiddens[-1][None].t())
    context_vector = torch.mm(alpha[None],source_hiddens[:,-1])

    probs = model.output_layer.forward(torch.cat((target_hiddens[-1][None],context_vector),dim=1).t())
    
    context_vector = torch.flatten(context_vector)
    
    return probs, target_hiddens, target_cells, context_vector, alpha

    """ Run the decoder AND the output layer for a single step.

    :param: prev_hidden: tensor of shape [L, hidden_dim] - the decoder's previous hidden state, denoted H^{dec}_{t-1}
                          in the assignment
    :param: source_hiddens: tensor of shape [source sentence length, L, hidden_dim] - the encoder's hidden states,
                            denoted H^{enc}_1 ... H^{enc}_T in the assignment
    :param: prev_context: tensor of shape [hidden_dim], denoted c_{t-1} in the assignment
    :param input: int - the word being inputted to the decoder.
                            during log-likelihood computation, this is y_{t-1}
                            during greedy decoding, this is yhat_{t-1}
    :param model: a Seq2SeqAttention model
    :return: (1) a tensor `probs` of shape [target_vocab_size], denoted p(y_t | x_1 ... x_S, y_1 .. y_{t-1})
             (2) a tensor `hidden` of shape [L, hidden_size], denoted H^{dec}_t in the assignment
             (3) a tensor `cell` of shape [L, cell_size], denoted C^{dec}_t in the assignment
             (4) a tensor `context` of shape [hidden_size], denoted c_t in the assignment
             (5) a tensor `attention_weights` of shape [source_sentence_length], denoted \alpha in the assignment
    """

    
    raise NotImplementedError()

def log_likelihood(source_sentence: List[int],
                   target_sentence: List[int],
                   model: Seq2SeqAttentionModel) -> torch.Tensor:

    log_like = 0
    source_hidden,source_cell = encode_all(source_sentence,model)
    
    target_hidden = torch.zeros([2,model.hidden_dim])
    target_cell = torch.zeros([2,model.hidden_dim])

    init_context_vector = torch.zeros([model.hidden_dim])


    for i in range(len(target_sentence)):
        if i==0:
            probs, target_hidden, target_cell, context_vector, alpha = decode(source_hidden[-1],source_cell[-1],source_hidden,init_context_vector,SOS_token,model)
            log_like += torch.log(probs[target_sentence[i]])             
        else:
            probs, target_hidden, target_cell, context_vector, alpha = decode(target_hidden,target_cell,source_hidden,context_vector,target_sentence[i-1],model)
            log_like += torch.log(probs[target_sentence[i]])
    return log_like


    """ Compute the log-likelihood for a (source_sentence, target_sentence) pair.

    :param source_sentence: the source sentence, as a list of words
    :param target_sentence: the target sentence, as a list of words
    :return: log-likelihood of the (source_sentence, target_sentence) pair
    """
    
    raise NotImplementedError()


def perplexity(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel) -> float:
    
    source_sentence = [x[0] for x in sentences]
    target_sentence = [x[1] for x in sentences]

    log_like = 0
    LL_total = 0
    count = 0
    log_like_list = []

    for i in range(len(sentences)):
        log_like = log_likelihood(source_sentence[i],target_sentence[i],model)
        log_like_list.append(log_like)
 
    for i in log_like_list:
        LL_total += i

    for sent in target_sentence:
        count += len(sent)
    perplexity = math.exp(-1 * LL_total/count)
    return perplexity
    
    """ Compute the perplexity of an entire dataset under a seq2seq model.  Refer to the write-up for the
    definition of perplexity.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: seq2seq attention model
    :return: perplexity of the translation
    """
    
    raise NotImplementedError()


def translate_greedy_search(source_sentence: List[int],
                            model: Seq2SeqAttentionModel, max_length=10) -> (List[int], torch.tensor):
    
    log_like = 0
    source_hidden,source_cell = encode_all(source_sentence,model)
    
    target_hidden = torch.zeros([2,model.hidden_dim])
    target_cell = torch.zeros([2,model.hidden_dim])

    init_context_vector = torch.zeros([model.hidden_dim])

    whole_sentence = []
    attention_matrix = []
    out = []

    for i in range(max_length): 

        if i==0:
            probs, target_hidden, target_cell, context_vector, alpha = decode(source_hidden[-1],source_cell[-1],source_hidden,init_context_vector,SOS_token,model)
            max_val,best_word = torch.max(probs,dim=0) 
        else:
            probs, target_hidden, target_cell, context_vector, alpha = decode(target_hidden,target_cell,source_hidden,context_vector,best_word,model)
            max_val,best_word = torch.max(probs,dim=0)
        whole_sentence.append(best_word)
        attention_matrix.append(alpha)
        attention_matrix_tensor = torch.stack(attention_matrix,dim=0)
        # print(whole_sentence[-1].tolist())
        out.append(whole_sentence[-1].tolist())
    return out,attention_matrix_tensor

    
    
    """ Translate a source sentence using greedy decoding.

    :param source_sentence: the source sentence, as a list of words
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the translated sentence as a list of ints
             (2) the attention matrix, a tensor of shape [target_sentence_length, source_sentence_length]

    """
    
    raise NotImplementedError()


def translate_beam_search(source_sentence: List[int], model: Seq2SeqAttentionModel,
                          beam_width: int, max_length=10) -> Tuple[List[int], float]:


    beam_width = 1
    # beam_width = 2
    # # beam_width = 4
    # beam_width = 8
    # source_sentence = source_sentence[0:20]

    # print(beam_width)
    # print(len(source_sentence))

    ans_sentences = []
    ans_probs = []
    cands = []
    source_hiddens, last_cell = encode_all(source_sentence, model)
    last_context = torch.zeros(source_hiddens.shape[-1])

    log_prob, last_hidden, last_cell, last_context, alpha = decode(source_hiddens[-1], last_cell[-1], source_hiddens, last_context, SOS_token, model)
    log_prob = torch.log(log_prob)
    
    ps = []
    sentences = []
    last_hiddens = []
    last_cells = []
    last_contexts = []
    tops, locs = log_prob.topk(beam_width)
    for i in range(len(locs)):
        sentences.append([locs[i]])
        ps.append(tops[i])
        last_hiddens.append(last_hidden)
        last_cells.append(last_cell)
        last_contexts.append(last_context)

    for counter in range(1, max_length):
        next_ps = []
        next_sentences = []
        next_hiddens = []
        next_cells = []
        next_contexts = []
        for i in range(len(sentences)):
            
            p, h, cell, context, _ = decode(last_hiddens[i], last_cells[i], source_hiddens, last_contexts[i], sentences[i][-1], model)
            tops, locs = torch.log(p).topk(beam_width)
            for j in range(len(locs)):
                next_ps.append(ps[i] + tops[j])
                next_sentences.append(sentences[i]+[locs[j]])
                next_hiddens.append(h)
                next_cells.append(cell)
                next_contexts.append(context)

        topps, toplocs = torch.topk(torch.Tensor(next_ps), beam_width)

        notends = []
        for i in toplocs:
            if next_sentences[i][-1] == EOS_token:
                ans_sentences.append(next_sentences[i])
                ans_probs.append(next_ps[i])
                beam_width -= 1
            else:
                notends.append(i)
        ps = [next_ps[i] for i in notends]
        sentences = [next_sentences[i] for i in notends]
        last_hiddens = [next_hiddens[i] for i in notends]
        last_contexts = [next_contexts[i] for i in notends]
        last_cells = [next_cells[i] for i in notends]

        if beam_width == 0: break


    for i in range(len(sentences)):
        ans_probs.append(ps[i])
        ans_sentences.append(sentences[i])

    topi = torch.argmax(torch.Tensor(ans_probs))

    out = []
    for i in sentences[topi]:
        out.append(i.item())
    
    return out, ans_probs[topi].detach().numpy()


    """ Translate a source sentence using beam search decoding.

    :param beam_width: the number of translation candidates to keep at each time step
    :param max_length: the maximum length that the target sentence could be
    :return: (1) the target sentence (translation),
             (2) sum of conditional log-likelihood of the translation, i.e., log p(target sentence|source sentence)
    """
    
    raise NotImplementedError()



def train_epoch(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
                epoch: int, print_every: int = 100, learning_rate: float = 0.0001, gradient_clip=5):
    """ Train the model for an epoch.

    :param sentences: list of (source_sentence, target_sentence) pairs
    :param model: a Seq2Seq model
    :param epoch: which epoch we're at
    """
    print("epoch\titer\tavg loss\telapsed secs")
    total_loss = 0
    start_time = time()
    optimizer = optim.Adam(model_params.values(), lr=learning_rate)
    for i, (source_sentence, target_sentence) in enumerate(sentences):
        optimizer.zero_grad()
        # print(i)
        theloss = -log_likelihood(source_sentence, target_sentence, model)
        total_loss += theloss
        theloss.backward()

        torch.nn.utils.clip_grad_norm_(model_params.values(), gradient_clip)

        optimizer.step()

        if i % print_every == 0:
            avg_loss = total_loss / print_every
            total_loss = 0
            elapsed_secs = time() - start_time
            print("{}\t{}\t{:.3f}\t{:.3f}".format(epoch, i, avg_loss, elapsed_secs))

    return model_params


def print_translations(sentences: List[Tuple[List[int], List[int]]], model: Seq2SeqAttentionModel,
                       vocab: Vocab, beam_width):
    """ Iterate through a dataset, printing (1) the source sentence, (2) the actual target sentence, and (3)
    the translation according to our model.

    :param sentences: a list of (source sentence, target sentence) pairs
    :param model: a Seq2Seq model
    :param vocab: a Vocab object
    """
    for (source_sentence, target_sentence) in sentences:
        if beam_width > 0:
            translation, _ = translate_beam_search(source_sentence, model, beam_width)
        else:
            translation, _ = translate_greedy_search(source_sentence, model)

        print("source sentence:" + " ".join([vocab.src.id2word[word] for word in source_sentence]))
        print("target sentence:" + " ".join([vocab.tgt.id2word[word] for word in target_sentence]))
        print("translation:\t" + " ".join([vocab.tgt.id2word[word] for word in translation]))
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Seq2Seq Homework Assignment')
    parser.add_argument("action", type=str,
                        choices=["train", "finetune", "train_perplexity", "test_perplexity",
                                 "print_train_translations", "print_test_translations", "visualize_attention"])
    parser.add_argument("--beam_width", type=int, default=-1,
                        help="number of translation candidates to keep at each time step. "
                             "this option is used for beam search translation (greedy search decoding is used by default).")
    parser.add_argument("--load_model", type=str,
                        help="path to saved model on disk.  if this arg is unset, the weights are initialized randomly")
    parser.add_argument("--save_model_prefix", type=str, help="prefix to save model with, if you're training")
    args = parser.parse_args()

    # load train/test data, and source/target vocabularies
    train_sentences, test_sentences, vocab = load_data()

    # load model weights (if path is specified) or else initialize weights randomly
    model_params = initialize_seq2seq_attention_params() if args.load_model is None \
        else torch.load(args.load_model)  # type: Dict[str, torch.Tensor]

    # build a Seq2SeqAttentionModel object
    model = build_seq2seq_attention_model(model_params)  # type: Seq2SeqAttentionModel

    if args.action == 'train':
        for epoch in range(10):
            train_epoch(train_sentences, model, epoch)
            torch.save(model_params, '{}_{}.pth'.format(args.save_model_prefix, epoch))
    elif args.action == 'finetune':
        train_epoch(train_sentences[:1000], model, 0, learning_rate=1e-5)
        torch.save(model_params, "{}.pth".format(args.save_model_prefix))
    elif args.action == "print_train_translations":
        print_translations(train_sentences, model, vocab, args.beam_width)
    elif args.action == "print_test_translations":
        print_translations(test_sentences, model, vocab, args.beam_width)
    elif args.action == "train_perplexity":
        print("perplexity: {}".format(perplexity(train_sentences[:1000], model)))
    elif args.action == "test_perplexity":
        print("perplexity: {}".format(perplexity(test_sentences, model)))
    elif args.action == "visualize_attention":
        from plotting import visualize_attention
        

        # visualize the attention matrix for the first 5 test set sentences
        for i in range(5):
            source_sentence = test_sentences[i][0]
            predicted_sentence, attention_matrix = translate_greedy_search(source_sentence, model)
            source_sentence_str = [ vocab.src.id2word[w] for w in source_sentence]
            predicted_sentence_str = [ vocab.tgt.id2word[w] for w in predicted_sentence]
            visualize_attention(source_sentence_str, predicted_sentence_str,
                                attention_matrix.detach().numpy(), "images/{}.png".format(i))
 
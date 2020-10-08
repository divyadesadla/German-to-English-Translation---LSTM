import unittest
from core import encode_all
from util import *
import torch
from seq2seq import translate_greedy_search, log_likelihood, decode
from seq2seq_attention import translate_greedy_search as translate_greedy_search_attention, \
    log_likelihood as log_likelihood_attention, decode as decode_attention, translate_beam_search, perplexity
from numpy.testing import assert_allclose
import pdb


TOLERANCE = 1e-4

model_params = initialize_seq2seq_params()
# seq2seq_model = build_seq2seq_model(model_params)
seq2seq_model = build_seq2seq_model(torch.load("pretrained/seq2seq.pth"))
seq2seq_attention_model = build_seq2seq_attention_model(torch.load("pretrained/seq2seq_attention.pth"))

# to run one test: python -m unittest tests.TestLSTM
# to run all tests: python -m unittest tests

class TestLSTM(unittest.TestCase):
    def test(self):
        data = torch.load("testLSTM.pth")
        hidden, cell = seq2seq_model.encoder.lstms[0].forward(data["input"], data["last_hidden"], data["last_cell"])
        assert_allclose(hidden.detach().numpy(), data["hidden"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(cell.detach().numpy(), data["cell"].detach().numpy(), atol=TOLERANCE)

class TestStackedLSTM(unittest.TestCase):
    def test(self):
        data = torch.load("test_stacked_LSTM.pth")
        hidden, cell = seq2seq_model.encoder.forward(data["input"], data["last_hidden"],data["last_cell"]) 
        assert_allclose(hidden.detach().numpy(), data["hidden"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(cell.detach().numpy(), data["cell"].detach().numpy(), atol=TOLERANCE)

class TestOutputLayer(unittest.TestCase):
    def test(self):
        data = torch.load("test_output.pth")
        output = seq2seq_model.output_layer.forward(data["input"])
        assert_allclose(output.detach().numpy(), data["output"].detach().numpy(), atol=TOLERANCE)

class TestEncodeAll(unittest.TestCase):
    def test(self):
        data = torch.load("test_encode_all.pth")
        source_hiddens, source_cells = encode_all(data["source_sentence"], seq2seq_model)
        assert_allclose(source_hiddens.detach().numpy(), data["source_hiddens"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(source_cells.detach().numpy(), data["source_cells"].detach().numpy(), atol=TOLERANCE)

class TestDecode(unittest.TestCase):
    def test(self):
        data = torch.load("test_decode.pth")
        out_probs, out_hidden, out_cell = decode(data["prev_hidden"], data["prev_cell"], data["input"], seq2seq_model)
        assert_allclose(out_probs.detach().numpy(), data["out_probs"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_hidden.detach().numpy(), data["out_hidden"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_cell.detach().numpy(), data["out_cell"].detach().numpy(), atol=TOLERANCE)


class TestLogLikelihood(unittest.TestCase):
    def test(self):
        data = torch.load("test_log_likelihood.pth")
        ll = log_likelihood(data["source_sentence"], data["target_sentence"], seq2seq_model)
        self.assertAlmostEqual(ll.item(), data["log_likelihood"].item(), places=4)


class TestTranslate(unittest.TestCase):
    def test(self):
        data = torch.load("test_translate.pth")
        target_sentence = translate_greedy_search(data["source_sentence"], seq2seq_model)
        self.assertEqual(target_sentence, data["target_sentence"])


class TestAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_attention.pth")
        attention_weights = seq2seq_attention_model.attention.forward(data["source_top_hiddens"],
                                                                      data["target_top_hidden"])
        assert_allclose(attention_weights.detach().numpy(), data["attention_weights"].detach().numpy(), atol=TOLERANCE)


class TestDecodeAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_decode_attention.pth")
        out_probs, out_hidden, out_cell, out_context, out_attention_weights = decode_attention(data["prev_hidden"],
                                                                                     data["prev_cell"],
                                                                                     data["source_hiddens"],
                                                                                     data["prev_context"], data["input"],
                                                                                     seq2seq_attention_model)
        assert_allclose(out_probs.detach().numpy(), data["out_probs"].detach().numpy(), atol=TOLERANCE),
        assert_allclose(out_hidden.detach().numpy(), data["out_hidden"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_cell.detach().numpy(), data["out_cell"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_context.detach().numpy(), data["out_context"].detach().numpy(), atol=TOLERANCE)
        assert_allclose(out_attention_weights.detach().numpy(),
                        data["out_attention_weights"].detach().numpy(), atol=TOLERANCE)


class TestLogLikelihoodAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_log_likelihood_attention.pth")
        ll = log_likelihood_attention(data["source_sentence"], data["target_sentence"], seq2seq_attention_model)
        assert_allclose(ll.item(), data["log_likelihood"].item(), atol=TOLERANCE)


class TestTranslateAttention(unittest.TestCase):
    def test(self):
        data = torch.load("test_translate_attention.pth")
        target_sentence, attention_matrix = translate_greedy_search_attention(data["source_sentence"],
                                                                              seq2seq_attention_model)
        self.assertEqual(target_sentence, data["target_sentence"])
        assert_allclose(attention_matrix.detach().numpy(), data["attention_matrix"].detach().numpy(), atol=TOLERANCE)


class TestTranslateBeamSearch(unittest.TestCase):
    def test(self):
        data = torch.load("test_translate_beam_search.pth")
        target_sentence, sum_log_likelihood = translate_beam_search(data["source_sentence"], seq2seq_attention_model,
                                                                    beam_width=4, max_length=10)
        self.assertEqual(target_sentence, data["target_sentence"])
        assert_allclose(sum_log_likelihood, data["sum_log_likelihood"], atol=TOLERANCE)

class TestPerplexity(unittest.TestCase):
    def test(self):
        data = torch.load("test_perplexity.pth")
        ppl = perplexity(data["sentences"], seq2seq_attention_model)
        assert_allclose(ppl, data["ppl"], atol=TOLERANCE)


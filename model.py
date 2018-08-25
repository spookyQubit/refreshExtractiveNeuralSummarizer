from config import FlagsClass

import torch.nn as nn
import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


FLAGS = FlagsClass()


def get_effective_number_of_samples(docs, weights):
    """
    :param docs: batch_size, max_doc_length, FLAGS.target_label_size
    :param weights: batch_size, max_doc_length
    :return: scalar
    """
    batch_size, num_sents, num_classes = docs.shape
    effective_number_of_samples = batch_size * num_sents
    if FLAGS.weighted_loss:
        weights = weights.view(-1)  # batch_size * max_doc_length
        weights = weights.type(t.cuda.FloatTensor) if FLAGS.use_cuda else weights.type(t.FloatTensor)
        effective_number_of_samples = weights.sum()

    return effective_number_of_samples.item()


class WordEmbedding(nn.Module):
    def __init__(self, word_embedding_array):
        super(WordEmbedding, self).__init__()

        word_embed_size = word_embedding_array.shape[1]
        weight_pad = Parameter(t.from_numpy(np.zeros((1, word_embed_size))).float(), requires_grad=False)
        weight_unk = Parameter(t.from_numpy(np.zeros((1, word_embed_size))).float(), requires_grad=True)
        weight_vocab = Parameter(t.from_numpy(word_embedding_array).float(), requires_grad=False)

        weight_all = t.cat([weight_pad, weight_unk, weight_vocab], 0)

        """
        With the current implementation, vocab as well as unk both have requires_grad=True.
        This is not the behavior in the paper where only unk has requires_grad=True.
        In pytorch, cannot find a way to have some index to have
        requires_grad=True and others requires_grad=False (except for padding_index).
        """
        self.all_embeddings = nn.Embedding(weight_all.shape[0], word_embed_size, padding_idx=0)
        self.all_embeddings.weight = Parameter(weight_all)  # Overrides the requires_grad set in Parameters above

    def forward(self, word_input):
        """
        :param word_input: [batch_size, seq_len] tensor of Long type
        :return: input embedding with shape of [batch_size, seq_len, word_embed_size]
        """
        return self.all_embeddings(word_input)


class SentEncoder(nn.Module):
    def __init__(self):
        super(SentEncoder, self).__init__()
        self.out_channels = FLAGS.out_channels
        self.in_channels = FLAGS.wordembed_size
        self.sentembedding_size = FLAGS.sentembedding_size
        self.kernel_widths = FLAGS.kernel_widths

        if self.sentembedding_size != (self.out_channels * len(self.kernel_widths)):
            raise ValueError("sent embed != out_chan * kW")

        self.kernels = [Parameter(t.Tensor(self.out_channels,
                                           self.in_channels,
                                           kW).normal_(0, 0.05)) for kW in self.kernel_widths]
        self._add_to_parameters(self.kernels, 'SentEncoderKernel')
        self.bias = Parameter(t.Tensor(self.out_channels).normal_(0, 0.05))

        self.lrn = nn.LocalResponseNorm(self.out_channels)

    def forward(self, sentences):
        """
        :param sentences: batch_size (each being a different sentence), each_sent_length, wordembed_size
        :return: batch_size, sentembedding_size
        """
        sentences = sentences.transpose(1, 2).contiguous()  # in_channel (i.e. word embed) has to be at 1
        xs = [F.relu(F.conv1d(sentences, kernel, bias=self.bias)) for kernel in self.kernels]  # [(batch_size, out_channels, k-h+1)] * len(self.kernels)
        xs = [F.max_pool1d(x, x.shape[2]) for x in xs]  # [(batch_size, out_channels, 1)] * len(self.kernels)
        xs = [self.lrn(x) for x in xs]  # [(batch_size, out_channels, 1)] * len(self.kernels)
        xs = [x.squeeze(2) for x in xs]  # [(batch_size, out_channels)] * len(self.kernels)
        xs = t.cat(xs, 1)  # batch_size, out_channels * len(self.kernels) == batch_size, sentembedding_size
        return xs

    def _add_to_parameters(self, parameters, name):
            for i, parameter in enumerate(parameters):
                self.register_parameter(name='{}-{}'.format(name, i), param=parameter)


class SentExtractor(nn.Module):
    def __init__(self):
        super(SentExtractor, self).__init__()

        self.input_size = FLAGS.sentembedding_size
        self.hidden_size = FLAGS.docembedding_size
        self.layers = FLAGS.rnn_layers
        self.bidirectional = FLAGS.bidirectional

        if self.layers != 1 or self.bidirectional:
            raise ValueError("Only {} layer(s) and uni-directional is supported at present".format(1))

        # In the paper, the extractor had LSTM
        # Using GRU here for simplicity
        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.layers,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.ll_output_size = FLAGS.target_label_size
        self.ll = nn.Linear(self.hidden_size, self.ll_output_size)

    def forward(self, sentence_embeddings, doc_embedding):
        """
        :param sentence_embeddings: batch_size, num_sents, FLAGS.sentembedding_size
        :param doc_embedding: batch_size, FLAGS.docembedding_size
        :return: logits. shape = batch_size, num_sents, FLAGS.target_label_size
        """

        [_, seq_len, _] = sentence_embeddings.size()

        # even after batch_first=True, the h0 follows [num_layers * num_directions, batch, hidden_size]
        doc_embedding = doc_embedding.unsqueeze(1).transpose(0, 1).contiguous()
        h_t, _ = self.rnn(sentence_embeddings, doc_embedding)  # batch, seq_len, self.hidden_size

        h_t = h_t.contiguous().view(-1, self.hidden_size)
        h_t = self.ll(h_t)
        h_t = h_t.view(-1, seq_len, self.ll_output_size)  # batch, seq_len, self.ll_output_size
        return h_t


class DocEncoder(nn.Module):
    def __init__(self):
        super(DocEncoder, self).__init__()

        self.input_size = FLAGS.sentembedding_size
        self.hidden_size = FLAGS.docembedding_size
        self.layers = FLAGS.rnn_layers
        self.bidirectional = FLAGS.bidirectional

        self.rnn = nn.LSTM(input_size=self.input_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)

    def forward(self, input):
        """
        :param input: batch_size, num_sents, FLAGS.sentembedding_size
        :return:
        """
        h_t, (h_n, _) = self.rnn(input)

        directions = 2 if self.bidirectional else 1
        h_n = h_n.view(-1, self.layers*directions, self.hidden_size)
        h_n = h_n.squeeze(1)
        return h_t, h_n


def get_cross_entropy(logits, weights, oracle_multiple, reward_multiple):
    """
    :param logits: batch_size, max_doc_length, FLAGS.target_label_size
    :param weights: batch_size, max_doc_length
    :param oracle_multiple: batch_size, 1, max_doc_length, FLAGS.target_label_size
    :param reward_multiple: batch_size, 1
    :return: cross_entropy. FloatTensor. shape = batch_size, 1, FLAGS.max_doc_length
    """
    batch_size, num_sents, target_label_size = logits.shape

    logits = logits.view(-1, FLAGS.target_label_size)  # batch_size*max_doc_length, target_label_size
    oracle_multiple = oracle_multiple.view(-1, FLAGS.target_label_size)  # batch_size*1*max_doc_length, target_label_size

    target = t.max(oracle_multiple, 1)[1]
    class_weights = t.from_numpy(np.array(FLAGS.class_weights))
    class_weights = class_weights.type(t.cuda.FloatTensor) if FLAGS.use_cuda else class_weights.type(t.FloatTensor)
    cross_entropy = F.cross_entropy(logits, target, weight=class_weights, reduce=False)

    if FLAGS.reinforcement:
        # reward_multiple : (batch_size, num_samples) -> (batch_size, num_samples, num_sents)
        # batch_size = 4
        # num_samples = 2
        # num_sentences = 3
        # [[0.8, 1.8], [0.7, 1.7], [0.6, 1.6], [0.5, 1.5]]
        #   --> [[[0.8, 0.8, 0.8], [1.8, 1.8, 1.8]],
        #         [[0,7, 0.7, 0.7], [1.7, 1.7, 1.7]],
        #         [[0.6, 0.6, 0.6], [1.6, 1.6, 1.6]],
        #         [[0.7, 0.7, 0.7], [1.7, 1.7, 1.7]]]

        reward_multiple = reward_multiple.view(-1)
        reward_multiple = reward_multiple.repeat(1, num_sents)
        reward_multiple = reward_multiple.view(num_sents, -1)
        reward_multiple = reward_multiple.transpose(0, 1).contiguous()
        reward_multiple = reward_multiple.view(batch_size, 1, num_sents)

        cross_entropy = cross_entropy.view(batch_size, 1, num_sents)
        cross_entropy = t.mul(cross_entropy, reward_multiple)
        cross_entropy = cross_entropy.view(-1)

    effective_number_of_samples = batch_size * num_sents
    if FLAGS.weighted_loss:
        weights = weights.view(-1)  # batch_size * max_doc_length
        weights = weights.type(t.cuda.FloatTensor) if FLAGS.use_cuda else weights.type(t.FloatTensor)
        effective_number_of_samples = weights.sum()
        cross_entropy = t.mul(cross_entropy, weights)

    return cross_entropy.sum()/effective_number_of_samples


def get_accuracy_metrics(logits, labels, weights):
    """
        :param logits: shape batch_size, num_sents, FLAGS.target_label_size
        :param labels: shape batch_size, num_sents, FLAGS.target_label_size
        :param weights: shape batch_size, num_sents
        :return: accuracy: scalar
                 precision: (num_classes)
                 recall: (num_classes)

        Example after logits/lables/weights are reshaped:
                       class 0 is the correct label ----                     --- class 1 is the correct label
                                                        |                    |
        labels: shape = (num_samples, num_classes) [[1, 0], [1, 0], [0, 1], [1, 0], [0, 1]]
        logits: shape = (num_samples, num_classes) [[1.3, 0.2], [0.8, -.1], [0.1, 0.3], [1.4, .2], [0.5, 0.1]]
        weights: shape = (num_samples) [1, 1, 1, 1, 0]
    """

    batch_size, num_sents, num_classes = logits.shape
    logits = logits.view(-1, num_classes)
    labels = labels.view(-1, num_classes)

    effective_number_of_samples = batch_size * num_sents

    precisions_per_class = np.zeros((num_classes))
    recalls_per_class = np.zeros((num_classes))
    correct_pred_per_class = np.zeros((num_classes))
    for c in range(num_classes):
        # Comments for c = 0
        true_labels = t.argmax(labels, dim=1)  # [0, 0, 1, 0]
        true_labels_for_current_class = true_labels == c  # [1, 1, 0, 1, 0]
        true_labels_for_current_class = true_labels_for_current_class.type(t.cuda.FloatTensor) if FLAGS.use_cuda \
            else true_labels_for_current_class.type(t.FloatTensor)

        predictions_for_class = t.argmax(logits, dim=1) == c  # [1, 1, 0, 1, 1]
        predictions_for_class = predictions_for_class.type(t.cuda.FloatTensor) if FLAGS.use_cuda \
            else predictions_for_class.type(t.FloatTensor)

        if FLAGS.weighted_loss:
            weights = weights.view(-1)  # batch_size * max_doc_length
            weights = weights.type(t.cuda.FloatTensor) if FLAGS.use_cuda else weights.type(t.FloatTensor)

            effective_number_of_samples = weights.sum()  # 4

            true_labels_for_current_class = t.mul(true_labels_for_current_class, weights)  # [1, 1, 0, 1, 0]
            predictions_for_class = t.mul(predictions_for_class, weights)  # [1, 1, 0, 1, 0]

        num_samples_for_class = true_labels_for_current_class.sum().float()  # 3
        predicted_samples_for_class = predictions_for_class.sum().float()  # 3

        true_positives_for_class = true_labels_for_current_class * predictions_for_class  # [1, 1, 0, 1, 0]
        true_positives_for_class = true_positives_for_class.sum().float()  # 3

        precision_for_class = true_positives_for_class / predicted_samples_for_class  # 1.0
        recall_for_class = true_positives_for_class / num_samples_for_class  # 1.0

        precisions_per_class[c] = precision_for_class
        recalls_per_class[c] = recall_for_class
        correct_pred_per_class[c] = true_positives_for_class

    accuracy = sum(correct_pred_per_class)/effective_number_of_samples

    return accuracy, precisions_per_class, recalls_per_class


class DocSummarizer(nn.Module):
    def __init__(self, word_embedding_array):
        super(DocSummarizer, self).__init__()
        self.word_embedding = WordEmbedding(word_embedding_array)
        self.sent_encoder = SentEncoder()
        self.doc_encoder = DocEncoder()
        self.sent_extractor = SentExtractor()

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, batch_docs):
        """
        :param batch_docs: shape = batch_size, max_doc_length + max_title_length + max_image_length, max_sent_length
        :return: logits: shape batch_size, max_doc_length + max_title_length + max_image_length, FLAGS.target_label_size
        """
        batch_size, num_sents, each_sent_length = batch_docs.shape

        # Get word embeddings
        # [batch_size, num_sents, each_sent_length]
        #     -->[batch_size, num_sents, each_sent_length, FLAGS.wordembed_size]
        batch_docs = batch_docs.view(-1, each_sent_length)  # TODO requires .contiguous()?
        batch_docs = self.word_embedding(batch_docs)
        batch_docs = batch_docs.view(batch_size,
                                     num_sents,
                                     each_sent_length,
                                     FLAGS.wordembed_size)  # TODO: requires .contiguous()?

        # Get sentence encoding
        # [batch_size, num_sents, each_sent_length, FLAGS.wordembed_size]
        #     --> [batch_size, num_sents, FLAGS.sentembed_size]
        batch_docs = batch_docs.view(-1,
                                     each_sent_length,
                                     FLAGS.wordembed_size)
        batch_docs = self.sent_encoder(batch_docs)  # batch_size * num_sents, FLAGS.sentembed_size
        batch_docs = batch_docs.view(batch_size, num_sents, FLAGS.sentembedding_size)  # TODO requires .contiguous()?

        # Get doc encoding
        # [batch_size, num_sents, FLAGS.sentembedding_size]
        #     --> [batch_size, num_sents, FLAGS.docembedding_size], [batch_size, FLAGS.docembedding_size]
        document_sents_enc = batch_docs  # We need to reverse
        doc_encoder_outputs, doc_encoder_state = self.doc_encoder(document_sents_enc)

        # Sent Extractor
        # [batch_size, num_sents, FLAGS.sentembedding_size], [batch_size, FLAGS.docembedding_size]
        #     --> [batch_size, num_sents, FLAGS.target_label_size]
        document_sents_ext = batch_docs
        logits = self.sent_extractor(document_sents_ext, doc_encoder_state)
        return logits

    def trainer_validator(self, optimizer, mode="training"):
        def _trainer_validator(data):

            # [batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple] = data
            data = [Variable(t.from_numpy(var)) for var in data]
            #data = [var.long() for var in data]
            data = [data[0].long(), data[1].long(), data[2].long(), data[3].long(), data[4].float()]
            data = [var.cuda() if FLAGS.use_cuda else var for var in data]
            [batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple] = data

            effective_num_samples = get_effective_number_of_samples(batch_docs, batch_weights)

            logits = self(batch_docs)
            cross_entropy = get_cross_entropy(logits, batch_weights, batch_oracle_multiple, batch_reward_multiple)
            loss = cross_entropy.mean()  # Mean should take acount the weights too! This mean should happen in get_cross_entropy
            accuracy, precision, recall = get_accuracy_metrics(logits, batch_label, batch_weights)

            optimizer.zero_grad()
            loss.backward()
            if mode == "training":
                optimizer.step()

            return loss, accuracy, precision, recall, effective_num_samples

        return _trainer_validator

import torch.nn as nn
import torch as t
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


"""
This file contains snippets whcih were used during development.
The code here is not needed/integrated in any way for any functionality of the final model.
"""


def test_embedding_unk():
    l = [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9], [1.7, 2.8, 3.9], [2.7, 3.8, 4.9]]
    word_embedding_array = np.asarray(l)
    word_embed_size = word_embedding_array.shape[1]

    weight_l = Parameter(t.from_numpy(word_embedding_array).float(), requires_grad=False)
    weight_unk = Parameter(t.from_numpy(np.ones((1, word_embed_size))).float(), requires_grad=True)
    weight_pad = Parameter(t.from_numpy(np.zeros((1, word_embed_size))).float(), requires_grad=False)

    all_weights = t.cat([weight_pad, weight_unk, weight_l], 0)
    print("all_weights = {}".format(all_weights))
    all_embeddings = nn.Embedding(all_weights.shape[0], word_embed_size, padding_idx=0)
    all_embeddings.weight = Parameter(all_weights)  # Overrides the requires_grad set in Parameters above

    """
    Even though requires_grad=False for weight_l, as far as all_embeddings.weight is concerned, the
    unk parameters in unk index get gradient.
    Only one index is allowed to be the pad_index. So, we cannot pass a list to pad_index.
    """

    print("all_embeddings.weight.requires_grad = {}".format(all_embeddings.weight.requires_grad))

    pad_index = 0
    inp = t.LongTensor([[pad_index, 1, 2]])
    out = all_embeddings(inp)
    out = out.mean()
    out.backward()
    print(all_embeddings.weight.grad)
    for pad_grad in all_embeddings.weight.grad[pad_index]:
        print(pad_grad.item())
        assert (pad_grad.item() == 0.0)


def test_local_response_norm():
    lrn = nn.LocalResponseNorm(3)
    signal_2d = t.zeros(2, 3, 4, 4)
    signal_2d[1, 2, :, :] = t.ones(4, 4)
    signal_2d[1, 1, :, :] = 0.5 * t.ones(4, 4)
    signal_2d[1, 0, :, :] = 0.1 * t.ones(4, 4)
    output_2d = lrn(signal_2d)

    print("signal_2d = {}".format(signal_2d))
    print("signal_2d.shape = {}".format(signal_2d.shape))
    print("output_2d = {}".format(output_2d))
    print("output_2d.shape = {}".format(output_2d.shape))
    print(signal_2d - output_2d)


def test_sent_embedding_max_pooling1d():
    num_sentences = 2
    each_sentence_length = 12
    word_emb_size = 20
    sentences = t.randn(num_sentences, each_sentence_length, word_emb_size)
    print("sentences.shape = {}".format(sentences.shape))

    out_channels = 7
    in_channels = word_emb_size
    kWs = [5, 6]  # Number of words convolved in one step
    kernels = [Parameter(t.Tensor(out_channels, in_channels, kW).normal_(0, 0.05)) for kW in kWs]
    sentences = sentences.transpose(1, 2)
    x = [F.tanh(F.conv1d(sentences, kernel)) for kernel in kernels]
    print("x[0].shape = {}".format(x[0].shape))
    print("x[1].shape = {}".format(x[1].shape))
    print("x = {}".format(x))
    x = [F.max_pool1d(x_, x_.shape[2]) for x_ in x]
    x = [x_.squeeze(2) for x_ in x]
    print("x = {}".format(x))
    print("x[0].shape = {}".format(x[0].shape))
    print("x[1].shape = {}".format(x[1].shape))

    x = t.cat(x, 1)
    print("x = {}".format(x))
    print("x.shape = {}".format(x.shape))


def test_sent_embedding():
    num_sentences = 2
    each_sentence_length = 12
    word_emb_size = 20
    sentences = t.randn(num_sentences, each_sentence_length, word_emb_size)
    print("sentences.shape = {}".format(sentences.shape))

    out_channels = 7
    in_channels = word_emb_size
    kWs = [5, 6]  # Number of words convolved in one step
    kernels = [Parameter(t.Tensor(out_channels, in_channels, kW).normal_(0, 0.05)) for kW in kWs]
    sentences = sentences.transpose(1, 2)
    x = [F.tanh(F.conv1d(sentences, kernel)) for kernel in kernels]
    print("x[0].shape = {}".format(x[0].shape))
    print("x[1].shape = {}".format(x[1].shape))
    print("x = {}".format(x))
    x = [x_.max(2)[0] for x_ in x]
    print("x = {}".format(x))
    print("x[0].shape = {}".format(x[0].shape))
    print("x[1].shape = {}".format(x[1].shape))

    x = t.cat(x, 1)
    print("x = {}".format(x))
    print("x.shape = {}".format(x.shape))


def test_accuracy():
    batch_size_ = 3
    max_doc_length_ = 5
    target_label_size_ = 2

    logits = t.randn(batch_size_, max_doc_length_, target_label_size_).type(t.FloatTensor)
    print("logits = {}".format(logits))

    labels = t.zeros(batch_size_, max_doc_length_, target_label_size_).type(t.FloatTensor)
    for b in range(batch_size_):
        for d in range(max_doc_length_):
            first_idx = np.random.randint(0, 2)
            labels[b, d, 0] = first_idx
            labels[b, d, 1] = 1 if first_idx == 0 else 0
    print("lables = {}".format(labels))

    weights = t.ones(batch_size_, max_doc_length_).type(t.FloatTensor)
    weights[:, int(max_doc_length_/2):] = 0
    print("weights = {}".format(weights))

    weights = weights.view(-1)
    logits = logits.view(-1, target_label_size_)
    labels = labels.view(-1, target_label_size_)
    print("weights = {}".format(weights))
    print("lables = {}".format(labels))
    print("logits = {}".format(logits))

    correct_preds = t.eq(t.argmax(labels, dim=1), t.argmax(logits, dim=1)).type(t.FloatTensor)
    print("correct_preds = {}".format(correct_preds))
    weighed_correct_preds = t.mul(correct_preds, weights)
    print("weighed_correct_preds = {}".format(weighed_correct_preds))
    mean_weighed_correct_preds = t.mean(weighed_correct_preds)
    print("mean_weighed_correct_preds = {}".format(mean_weighed_correct_preds))

    sum_of_weights = weights.sum()
    weighed_averaged_correct_preds = weighed_correct_preds.sum()/sum_of_weights
    print("weighed_averaged_correct_preds = {}".format(weighed_averaged_correct_preds))



def test_precision():
    num_samples = 5
    num_classes = 2
    logits = np.zeros((num_samples, num_classes))
    logits[0] = [0.8, 0.2]
    logits[1] = [0.7, 0.3]
    logits[2] = [0.75, 0.25]
    logits[3] = [0.1, 0.9]
    logits[4] = [0.6, 0.4]
    logits = t.from_numpy(logits).float()
    print("logits = {}".format(logits))
    print("logits.shape = {}".format(logits.shape))
    print("t.argmax(logits, dim=1) = {}".format(t.argmax(logits, dim=1)))
    pred = t.argmax(logits, dim=1)

    labels = np.zeros((num_samples, num_classes))
    labels[0] = [1, 0]
    labels[1] = [1, 0]
    labels[2] = [1, 0]
    labels[3] = [0, 1]
    labels[4] = [0, 1]
    labels = t.from_numpy(labels).float()
    print("labels = {}".format(labels))
    print("labels.shape = {}".format(labels.shape))
    print("t.argmax(labels, dim=1) = {}".format(t.argmax(labels, dim=1)))
    #labels = t.argmax(labels, dim=1)

    weights = np.zeros(num_samples)
    weights[0] = 0
    weights[1] = 1
    weights[2] = 1
    weights[3] = 1
    weights[4] = 1
    weights = t.from_numpy(weights).float()
    accuracy_, precision_, recall_ = get_accuracy_metrics(labels, logits, weights)
    print("accuracy_ = {}, precision_ = {}, recall_ = {}".format(accuracy_, precision_, recall_))

    precisions_per_class = np.zeros((num_classes))
    recalls_per_class = np.zeros((num_classes))
    for c in range(num_classes):
        true_labels = t.argmax(labels, dim=1)
        true_labels_for_current_class = true_labels==c
        print("true_labels_for_current_class = {}".format(true_labels_for_current_class))

        predictions_for_class = t.argmax(logits, dim=1)==c
        print("predictions_for_class = {}".format(predictions_for_class))

        num_samples_for_class = true_labels_for_current_class.sum().float()
        predicted_samples_for_class = predictions_for_class.sum().float()

        true_positives_for_class = true_labels_for_current_class * predictions_for_class
        true_positives_for_class = true_positives_for_class.sum().float()
        print("true_positives_for_class = {}".format(true_positives_for_class))

        precision_for_class = true_positives_for_class / predicted_samples_for_class
        recall_for_class = true_positives_for_class / num_samples_for_class
        print("precision_for_class = {}, recall = {}".format(precision_for_class, recall_for_class))

        precisions_per_class[c] = precision_for_class
        recalls_per_class[c] = recall_for_class





    """
    #t.eq(t.argmax(labels, dim=1), t.argmax(logits, dim=1))
    negative_samples = t.eq(labels, 1).sum()
    positive_samples = t.eq(labels, 0).sum()
    print("negative_samples = {}".format(negative_samples))
    true_negatives = labels * pred
    true_negatives = true_negatives.sum()
    print("true_positive = {}".format(true_negatives))
    print("correct_preds = {}".format(t.eq(labels, pred).sum()))
    """

def test_class_weight_ce():
    labels = np.array([[1, 0], [1, 0], [0, 1]])
    logits = np.array([[0.5, 0.5], [2.4, 0.1], [0.4, 2.1]])
    print("labels = {}".format(labels))
    print("logits = {}".format(logits))

    labels = t.from_numpy(labels).long()
    logits = t.from_numpy(logits).float()
    print("labels = {}".format(labels))
    print("logits = {}".format(logits))

    target = t.max(labels, 1)[1]

    class_weights = t.from_numpy(np.array([10, 1])).float()

    ce = F.cross_entropy(weight=class_weights, input=logits, target=target, reduce=False)
    print("ce = {}".format(ce))


def test_reward_resizing():

    batch_size = 4
    num_samples = 2
    num_sentences = 3
    # r : (batch_size, num_samples) -> (batch_size, num_samples, num_sentences)
    # batch_size = 4
    # num_samples = 2
    # num_sentences = 3
    # [[0.8, 1.8], [0.7, 1.7], [0.6, 1.6], [0.5, 1.5]]
    #   --> [[[0.8, 0.8, 0.8], [1.8, 1.8, 1.8]],
    #         [[0,7, 0.7, 0.7], [1.7, 1.7, 1.7]],
    #         [[0.6, 0.6, 0.6], [1.6, 1.6, 1.6]],
    #         [[0.7, 0.7, 0.7], [1.7, 1.7, 1.7]]]

    r = np.array([[0.8, 1.8], [0.7, 1.7], [0.6, 1.6], [0.5, 1.5]])

    r = t.from_numpy(r).float()
    print("r.shape = {}".format(r.shape))

    r = r.view(-1)
    print("r.shape = {}".format(r.shape))  # [0.8, 1.8, 0.7, 1.7, 0.6, 1.6, 0.5, 1.5]
    print("r = {}".format(r))

    r = r.repeat(1, num_sentences)
    print("r.shape = {}".format(r.shape)) # [0.8, 1.8, 0.7, 1.7, 0.6, 1.6, 0.5, 1.5, 0.8, 1.8, 0.7 ....]
    print("r = {}".format(r))

    r = r.view(num_sentences, -1)
    print("r.shape = {}".format(r.shape))
    print("r = {}".format(r))

    r = r.transpose(0, 1).contiguous()
    print("r.shape = {}".format(r.shape))
    print("r = {}".format(r))

    r = r.view(batch_size, num_samples, num_sentences)
    print("r.shape = {}".format(r.shape))
    print("r = {}".format(r))


if __name__ == "__main__":
    #test_local_response_norm()
    #test_sent_embedding()
    #test_sent_embedding_max_pooling1d()
    #test_accuracy()
    #test_precision()
    #test_class_weight_ce()
    test_reward_resizing()

    print("Done running tests in model.py.")
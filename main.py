from data_utils import prepare_vocab_embeddingdict
from data_utils import BatchLoader
from model import DocSummarizer
from torch.optim import Adam
from config import FlagsClass
import torch as t
import time
import logging


FLAGS = FlagsClass()


def get_logger():
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y:%H:%M:%S',
                        filename=FLAGS.log_file,
                        level=logging.INFO)
    logger = logging.getLogger(FLAGS.logger)
    return logger


def log_config(logger):
    config = ["\n*** Config ***"]
    for k, v in vars(FLAGS).items():
        config.append("{0} : {1}".format(k, v))
    config.append("\n")
    logger.info("\n".join(config))


def validate(batch_loader_v, validator, logger):
    batch_loader_v.shuffle()

    total_loss_v = 0.0
    total_accuracy_v = 0.0
    total_precision_v = 0.0
    total_recall_v = 0.0
    total_num_of_eval_labels_v = 0.0
    for data_v in batch_loader_v.next_batch(FLAGS.batch_size):
        loss_v, accuracy_v, precision_v, recall_v, num_of_eval_labels_v = validator(data_v)

        total_loss_v += (loss_v * num_of_eval_labels_v)
        total_accuracy_v += (accuracy_v * num_of_eval_labels_v)
        total_precision_v += (precision_v * num_of_eval_labels_v)
        total_recall_v += (recall_v * num_of_eval_labels_v)
        total_num_of_eval_labels_v += num_of_eval_labels_v

    total_loss_v = total_loss_v / total_num_of_eval_labels_v
    total_accuracy_v = total_accuracy_v / total_num_of_eval_labels_v
    total_precision_v = total_precision_v / total_num_of_eval_labels_v
    total_recall_v = total_recall_v / total_num_of_eval_labels_v
    logger.info("Validation : loss = {0:.2f} : "
                "accuracy = {1:.2f}, "
                "precision = {2}, "
                "recall = {3}, ".format(total_loss_v, total_accuracy_v, total_precision_v, total_recall_v))


def train():

    logger = get_logger()
    log_config(logger)

    t.manual_seed(FLAGS.torch_manual_seed)

    # Prepare data for training
    logger.info("Prepare vocab dict and read pretrained word embeddings ...")
    vocab_dict, word_embedding_array = prepare_vocab_embeddingdict(logger)

    batch_loader_t = BatchLoader(vocab_dict, logger=logger, data_type="training")
    batch_loader_v = BatchLoader(vocab_dict, logger=logger, data_type="validation")

    model = DocSummarizer(word_embedding_array)
    if FLAGS.use_trained:
        logger.info("Using trained model ...")
        model.load_state_dict(t.load('trainedRefreshModel'))

    optimizer = Adam(model.learnable_parameters(), FLAGS.learning_rate)
    trainer = model.trainer_validator(optimizer, "training")
    validator = model.trainer_validator(optimizer, "validation")

    if FLAGS.use_cuda:
        model.cuda()

    for epoch in range(FLAGS.num_epochs):

        batch_loader_t.shuffle()

        batch_number = 0
        time_start = time.time()
        logger.info("Start time = {}".format(time_start))
        for data in batch_loader_t.next_batch(FLAGS.batch_size):
            loss_t, accuracy_t, precision_t, recall_t, _ = trainer(data)
            if batch_number % 50 == 0:
                logger.info("Processed {0:.2f} samples, "
                            "loss = {1:.2f}, "
                            "accuracy = {2:.2f}, "
                            "precision = {3}, "
                            "recall = {4}".format((batch_number+1) * FLAGS.batch_size,
                                                  loss_t, accuracy_t, precision_t, recall_t))

            if batch_number % 500 == 0:
                validate(batch_loader_v, validator, logger)

            batch_number += 1

        logger.info("Saving trained model")
        t.save(model.state_dict(), 'trainedRefreshModel')

        logger.info("Took {} secs to run an epoch".format(time.time() - time_start))


def main():
    train()


if __name__ == "__main__":
    main()

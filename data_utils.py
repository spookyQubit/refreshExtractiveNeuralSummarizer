from config import FlagsClass
import numpy as np
import random

# Special IDs
PAD_ID = 0
UNK_ID = 1

FLAGS = FlagsClass()


def _print_line(line, vocab_dict):
    id2word = {v: k for k, v in vocab_dict.items()}
    items = 0
    for k, v in id2word.items():
        print("{}: {}".format(k, v))
        items += 1
        if items == 100:
            break

    debug_first_line = []
    print("line = {}".format(line))
    for word_id in line:
        word_id = int(word_id)
        print("word_id = {}".format(word_id))
        print("id2word[word_id] = {}".format(id2word[word_id]))
        if word_id in id2word:
            debug_first_line.append(id2word[word_id])
        else:
            debug_first_line.append(id2word[UNK_ID])
    print("debug_first_line = {}".format(debug_first_line))


def prepare_vocab_embeddingdict(logger):
    # Numpy dtype
    dtype = np.float16 if FLAGS.use_fp16 else np.float32

    vocab_dict = {}
    word_embedding_array = []

    # Add _PAD, _UNK
    vocab_dict["_PAD"] = PAD_ID
    vocab_dict["_UNK"] = UNK_ID

    # Read word embedding file
    wordembed_filename = FLAGS.pretrained_wordembedding
    logger.info("Reading pretrained word embeddings file: %s" % wordembed_filename)

    embed_line = ""
    with open(wordembed_filename, "r") as fembedd:
        for linecount, line in enumerate(fembedd):
            if linecount == 0:
                vocabsize = int(line.split()[0])
                word_embedding_array = np.empty((vocabsize, FLAGS.wordembed_size), dtype)

            else:
                linedata = line.split()
                vocab_dict[linedata[0]] = linecount + 1  # because 0, 1 are reserved for _UNK, _PAD
                embeddata = [float(item) for item in linedata[1:]][0:FLAGS.wordembed_size]
                word_embedding_array[linecount - 1] = embeddata

            if linecount%100000 == 0:
                logger.info("linecount = {}".format(linecount))

        logger.info("Done reading pre-trained word embedding of shape {}".format(word_embedding_array.shape))

    logger.info("Size of vocab: %d (_PAD:0, _UNK:1)" % len(vocab_dict))
    vocabfilename = FLAGS.train_dir + "/vocab.txt"
    logger.info("Writing vocab file: %s" % vocabfilename)
    with open(vocabfilename, "w") as foutput:
        vocab_list = [(vocab_dict[key], key) for key in vocab_dict.keys()]
        vocab_list.sort()
        vocab_list = [item[1] for item in vocab_list]
        foutput.write("\n".join(vocab_list) + "\n")

    return vocab_dict, word_embedding_array


class Data:
    def __init__(self, vocab_dict, data_type, logger):
        self.logger = logger
        self.filenames = []
        self.docs = []
        self.titles = []
        self.images = []
        self.labels = []
        self.weights = []
        self.rewards = []
        self.fileindices = []

        self.data_type = data_type

        # populate the data
        self.populate_data(vocab_dict, data_type)

        # write to files
        self.write_to_files(data_type)

    def __len__(self):
        return len(self.fileindices)

    def shuffle_fileindices(self):
        random.shuffle(self.fileindices)

    def get_batch(self, startidx, endidx):
        dtype = np.float32

        def process_to_chop_pads(ordids, requiredsize):
            if len(ordids) >= requiredsize:
                return ordids[:requiredsize]
            else:
                padids = [PAD_ID] * (requiredsize - len(ordids))
                return ordids + padids

        # For training, (endidx-startidx)=FLAG.batch_size
        # For others, it is as specified
        batch_docnames = np.empty((endidx-startidx), dtype="S60")  # File ID "cnn-"/"dailymail-"
        batch_docs = np.empty(((endidx-startidx), (FLAGS.max_doc_length+FLAGS.max_image_length+FLAGS.max_title_length), FLAGS.max_sent_length), dtype="int32")
        batch_label = np.empty(((endidx-startidx), FLAGS.max_doc_length, FLAGS.target_label_size), dtype=dtype)
        batch_weights = np.empty(((endidx-startidx), FLAGS.max_doc_length), dtype=dtype)
        batch_oracle_multiple = np.empty(((endidx-startidx), 1, FLAGS.max_doc_length, FLAGS.target_label_size), dtype=dtype)
        batch_reward_multiple = np.empty(((endidx - startidx), 1), dtype=dtype)

        for batch_idx, fileindex in enumerate(self.fileindices[startidx:endidx]):
            # Document name
            batch_docnames[batch_idx] = self.filenames[fileindex]

            # Document
            doc_wordids = []  # [FLAGS.max_doc_length+FLAGS.max_image_length+FLAGS.max_title_length, FLAGS.max_sent_length]
            for idx in range(FLAGS.max_doc_length):
                thissent = []
                if idx < len(self.docs[fileindex]):
                    thissent = self.docs[fileindex][idx][:]
                thissent = process_to_chop_pads(thissent, FLAGS.max_sent_length)  # FLAGS.max_sent_length
                doc_wordids.append(thissent)
            for idx in range(FLAGS.max_title_length):
                thissent = []
                if idx < len(self.titles[fileindex]):
                    thissent = self.titles[fileindex][idx][:]
                thissent = process_to_chop_pads(thissent, FLAGS.max_sent_length)  # FLAGS.max_sent_length
                doc_wordids.append(thissent)
            for idx in range(FLAGS.max_image_length):
                thissent = []
                if idx < len(self.images[fileindex]):
                    thissent = self.images[fileindex][idx][:]
                thissent = process_to_chop_pads(thissent, FLAGS.max_sent_length)  # FLAGS.max_sent_length
                doc_wordids.append(thissent)
            batch_docs[batch_idx] = np.array(doc_wordids[:], dtype="int32")

            # Labels: Select the single best, i.e. the first
            labels_vecs = [[1,0] if (item in self.labels[fileindex][0]) else [0, 1] for item in range(FLAGS.max_doc_length)]
            batch_label[batch_idx] = np.array(labels_vecs[:], dtype="int32")

            # Weights
            weights = process_to_chop_pads(self.weights[fileindex][:], FLAGS.max_doc_length)
            batch_weights[batch_idx] = np.array(weights[:], dtype=dtype)

            # Multiple label and rewards
            labels_set = []  # FLAG.num_sample_rollout, FLAGS.max_doc_length, FLAGS.target_label_size
            reward_set = []  # FLAG.num_sample_rollout, FLAGS.max_doc_length, FLAGS.target_label_size
            for idx in range(FLAGS.num_sample_rollout):
                thislabels = []
                if idx < len(self.labels[fileindex]):
                    thislabels = [[1, 0] if (item in self.labels[fileindex][idx]) else [0, 1] for item in range(FLAGS.max_doc_length)]
                    reward_set.append(self.rewards[fileindex][idx])
                else:
                    # Simply copy the best one
                    thislabels = [[1, 0] if (item in self.labels[fileindex][0]) else [0, 1] for item in range(FLAGS.max_doc_length)]
                    reward_set.append(self.rewards[fileindex][0])
                labels_set.append(thislabels)
            randidx_oracle = random.randint(0, (FLAGS.num_sample_rollout - 1))
            batch_oracle_multiple[batch_idx][0] = np.array(labels_set[randidx_oracle][:], dtype=dtype)
            batch_reward_multiple[batch_idx] = np.array([reward_set[randidx_oracle]], dtype=dtype)

        return batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple

    def write_to_files(self, data_type):
        full_data_file_prefix = FLAGS.train_dir + "/" + FLAGS.data_mode + "." + data_type
        self.logger.info("Writing files with prefix (.doc, .title, .image, .label.jp-org): %s" % full_data_file_prefix)

        ffilenames = open(full_data_file_prefix + ".filename", "w")
        fdoc = open(full_data_file_prefix + ".doc", "w")
        ftitle = open(full_data_file_prefix + ".title", "w")
        fimage = open(full_data_file_prefix + ".image", "w")
        flabel = open(full_data_file_prefix + ".label", "w")
        fweight = open(full_data_file_prefix + ".weight", "w")
        freward = open(full_data_file_prefix + ".reward", "w")

        for filename, doc, title, image, label, weight, reward in zip(self.filenames, self.docs, self.titles,
                                                                      self.images, self.labels, self.weights,
                                                                      self.rewards):
            ffilenames.write(filename + "\n")
            fdoc.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in doc]) + "\n\n")
            ftitle.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in title]) + "\n\n")
            fimage.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in image]) + "\n\n")
            flabel.write("\n".join([" ".join([str(item) for item in itemlist]) for itemlist in label]) + "\n\n")
            fweight.write(" ".join([str(item) for item in weight]) + "\n")
            freward.write(" ".join([str(item) for item in reward]) + "\n")

        ffilenames.close()
        fdoc.close()
        ftitle.close()
        fimage.close()
        flabel.close()
        fweight.close()
        freward.close()

        return

    def populate_data(self, vocab_dict, data_type):

        full_data_file_prefix = FLAGS.preprocessed_data_dir + "/" + FLAGS.data_mode + "." + data_type
        self.logger.info("Data file prefix (.doc, .title, .image, .label.jp-org): %s" % full_data_file_prefix)

        # Process doc, title, image and label
        doc_data_list = open(full_data_file_prefix+".doc").read().strip().split("\n\n")
        title_data_list = open(full_data_file_prefix + ".title").read().strip().split("\n\n")
        image_data_list = open(full_data_file_prefix + ".image").read().strip().split("\n\n")
        label_data_list = open(full_data_file_prefix + ".label.multipleoracle").read().strip().split("\n\n")

        self.logger.info("Preparing data per model requirement")
        doccount = 0
        for doc_data, title_data, image_data, label_data in zip(doc_data_list, title_data_list, image_data_list, label_data_list):

            doc_lines = doc_data.strip().split("\n")
            title_lines = title_data.strip().split("\n")
            image_lines = image_data.strip().split("\n")
            label_lines = label_data.strip().split("\n")

            filename = doc_lines[0].strip()
            if ((filename == title_lines[0].strip()) and
                    (filename == image_lines[0].strip()) and
                    (filename == label_lines[0].strip())):
                self.filenames.append(filename)

                # Doc
                thisdoc = []
                for line in doc_lines[1:FLAGS.max_doc_length+1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thisdoc.append(thissent)
                self.docs.append(thisdoc)

                # Title
                thistitle = []
                for line in title_lines[1:FLAGS.max_title_length+1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thistitle.append(thissent)
                self.titles.append(thistitle)

                # Image
                thisimage = []
                for line in image_lines[1:FLAGS.max_image_length + 1]:
                    thissent = [int(item) for item in line.strip().split()]
                    thisimage.append(thissent)
                self.images.append(thisimage)

                # Weights
                originaldoclen = int(label_lines[1].strip())
                thisweight = [1 for item in range(originaldoclen)][:FLAGS.max_doc_length]
                self.weights.append(thisweight)

                # Labels (multiple oracles and preestimated rewards)
                thislabel = []
                thisreward = []
                for line in label_lines[2:FLAGS.num_sample_rollout + 2]:
                    thislabel.append([int(item) for item in line.split()[:-1]])
                    thisreward.append(float(line.split()[-1]))
                self.labels.append(thislabel)
                self.rewards.append(thisreward)

                #line = label_lines[1]
                #self._print_line(line, vocab_dict)
            else:
                self.logger.error("Some problem with %s.* files. Exiting!" % full_data_file_prefix)
                exit(1)

            if doccount % 10000 == 0:
                self.logger.info("%d ..." % doccount)

            doccount += 1

        # Set file indices
        self.fileindices = list(range(len(self.filenames)))
        self.logger.info("Read {} docs".format(len(self.filenames)))


def prepare_news_data(vocab_dict, logger, data_type="training"):
    data = Data(vocab_dict, data_type, logger=logger)
    return data


class BatchLoader:
    def __init__(self, vocab_dict, logger, data_type="training"):
        self.data = prepare_news_data(vocab_dict, logger, data_type)

    def shuffle(self):
        self.data.shuffle_fileindices()

    def get_all_data(self):
        return self.data.get_batch(startidx=0, endidx=len(self.data))

    def get_batch(self, startidx, endidx):
        return self.data.get_batch(startidx, endidx)

    def next_batch(self, batch_size):
        for startidx in range(0, len(self.data), batch_size):
            #batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple = self.data.get_batch(startidx, startidx+batch_size)
            #yield (batch_docs, batch_label, batch_weights, batch_oracle_multiple, batch_reward_multiple)
            #yield self.data[startidx: startidx + batch_size]
            yield self.data.get_batch(startidx, startidx + batch_size)

if __name__ == "__main__":

    print("main pf data_utils")
    b = BatchLoader([0], "validation")
    print(len(b.get_all_data()[0]))
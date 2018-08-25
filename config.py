class FlagsClass:
    def __init__(self):
        self.use_fp16 = False
        self.use_trained = False
        self.torch_manual_seed = 42
        self.log_file = "refresh.log"
        self.logger = "Refresh"
        self.pretrained_wordembedding = "data/1-billion-word-language-modeling-benchmark-r13output.word2vec.vec"
        self.wordembed_size = 200 # has to be <= 200
        self.train_dir = "train_dir"
        self.preprocessed_data_dir = "data/preprocessed-input-directory"
        self.data_mode = "cnn"
        self.max_doc_length = 110  # max sents to consider  TODO 110
        self.max_title_length = 0  # max top titles to consider
        self.max_image_length = 0  # max image captions to consider
        self.max_sent_length = 100  # max words in a sent  TODO 100
        self.target_label_size = 2  # Size of target label (0/1)
        self.num_sample_rollout = 10  # Number of Multiple Oracles Used.
        self.num_epochs = 20
        self.batch_size = 20
        self.use_cuda = True
        self.out_channels = 50
        self.sentembedding_size = 350
        self.kernel_widths = [1, 2, 3, 4, 5, 6, 7]
        self.docembedding_size = 600
        self.rnn_layers = 1
        self.bidirectional = False
        self.weighted_loss = True
        self.learning_rate = 0.001
        self.class_weights = [5, 1]
        self.reinforcement = False

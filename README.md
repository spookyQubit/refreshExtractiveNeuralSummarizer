## Extractive summarization
Extractive summarization is a task of selecting the most summary-worthy sentences from a document. This differs from abstractive summarization where the summary is paraphrased and not extracted verbatim. Even in many abstractive summarization, extracting relevant sentences is an intermediary step before paraphrasing, thus highlighting the importance of the extractive task.

Traditionally, extracting sentences were carried out using graph based or count based techniques such as TextRank, LexRank and SumBasic. Now, neural network architectures, powered by curated large scale datasets needed to train them, have empirically pushed the state-of-the-art baselines on many NLP tasks including extractive summarization.

This repo is my attempt to learn more about this fascinating area of summarization than just reading research papers. Also, it is an opportunity to sharpen my pytorch skills.  

### Refresh
This is an unofficial pytorch implementation of [Refresh: Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf). The original code implementation in tensorflow can be found at [EdinburghNLP/Refresh](https://github.com/EdinburghNLP/Refresh).


To run, follow the instructions in [EdinburghNLP/Refresh](https://github.com/EdinburghNLP/Refresh) to download the dataset. All the hyperparameters can be set in config.py. The code has been tested only with the CNN dataset. 
```
Running the code: python main.py
```
To run the full experiment for 20 epochs (similar to the original tensorflow implementation, the full experiment takes around 12 hours to run), I used the following AWS credentials:
```
Instance: g3.4xlarge 
AMI: Deep Learning AMI (Ubuntu) Version 9.0. 
Env: pytorch_p36 (source activate pytorch_p36)
```

The cross_entropy loss class_weights in the original implementation were implicitly set to [1, 1]. However, the nature of the sentence extraction task involves highly skewed class statistics because the number of positive lables are rare, indicating summary-worthy sentences are far less frequent than non summary-worthy sentences. To account for this class imbalance, I changed the weight parameter in F.cross_entropy (set in class_weights parameter in config.py) to be different than [1, 1].

### TODO
- [ ] Test data
- [ ] Embedding requires_grad=False except for UNK.

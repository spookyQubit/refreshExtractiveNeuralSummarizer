## Extractive summarization

This is an unofficial pytorch implementation of [Refresh: Ranking Sentences for Extractive Summarization with Reinforcement Learning](https://arxiv.org/pdf/1802.08636.pdf). The original code implementation in tensorflow can be found in [EdinburghNLP/Refresh](https://github.com/EdinburghNLP/Refresh).


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

The nature of the sentence extraction task involves higly skewed class statistics since the number of positive lables indicating summary-worthy sentences is far less than non summary-worthy sentences. In order to tackle this imbalance, one can set the class_weights parameter in config.py to be different than [1, 1].

### TODO
- [ ] Test data
- [ ] Embedding requires_grad=False except for UNK.

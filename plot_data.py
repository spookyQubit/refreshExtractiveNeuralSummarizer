import matplotlib.pyplot as plt

trainin_dir = "./cleaning_log/with_rewards"
files = ["accuracy", "loss", "precision", "recall"]


# loss
plt.subplot(221)
filename = trainin_dir + "/" + "training_loss"
data = [float(line.rstrip('\n')) for line in open(filename)]
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Loss')
plt.plot(data)

# accuracy
plt.subplot(222)
filename = trainin_dir + "/" + "training_accuracy"
data = [float(line.rstrip('\n')) for line in open(filename)]
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Accuracy')
plt.plot(data)

# precision
plt.subplot(223)
filename = trainin_dir + "/" + "training_precision"
data = [float(line.rstrip('\n')) for line in open(filename)]
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Precision')
plt.plot(data)

# recall
plt.subplot(224)
filename = trainin_dir + "/" + "training_recall"
data = [float(line.rstrip('\n')) for line in open(filename)]
axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Recall')
plt.plot(data)

plt.show()
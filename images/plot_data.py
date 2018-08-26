import matplotlib.pyplot as plt

with_rewards_dir = "./cleaning_log/with_rewards"
without_rewards_dir = "./cleaning_log/without_rewards"
files = ["accuracy", "loss", "precision", "recall"]


# loss
plt.subplot(221)

filename = with_rewards_dir + "/" + "training_loss"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

filename = without_rewards_dir + "/" + "training_loss"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Loss')


# accuracy
plt.subplot(222)
filename = with_rewards_dir + "/" + "training_accuracy"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

filename = without_rewards_dir + "/" + "training_accuracy"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Accuracy')


# precision
plt.subplot(223)
filename = with_rewards_dir + "/" + "training_precision"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

filename = without_rewards_dir + "/" + "training_precision"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Precision')


# recall
plt.subplot(224)
filename = with_rewards_dir + "/" + "training_recall"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

filename = without_rewards_dir + "/" + "training_recall"
data = [float(line.rstrip('\n')) for line in open(filename)]
plt.plot(data)

axes = plt.gca()
axes.set_ylim([0, 1])
plt.title('Recall')


plt.show()
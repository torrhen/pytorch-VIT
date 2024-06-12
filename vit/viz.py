import matplotlib.pyplot as plt

def plot_loss(results: dict):
  train_loss = results['train_loss']
  val_loss = results['val_loss']

  epochs = [(i + 1) for i in range(len(train_loss))]
  _, ax = plt.subplots(figsize=(8, 8))
  ax.plot(epochs, train_loss, label='Training loss')
  ax.plot(epochs, val_loss, label='Validation loss')
  ax.set_title("ViT loss on CIFAR10")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  plt.legend()
  plt.savefig("loss.png")


def plot_accuracy(results: dict):
  train_accuracy = results['train_accuracy']
  val_accuracy = results['val_accuracy']

  epochs = [(i + 1) for i in range(len(train_accuracy))]
  _, ax = plt.subplots(figsize=(8, 8))
  ax.plot(epochs, train_accuracy, label='Training accuracy')
  ax.plot(epochs, val_accuracy, label='Validation accuracy')
  ax.set_title("ViT accuracy on CIFAR10")
  ax.set_xlabel("Epochs")
  ax.set_ylabel("Accuracy")
  plt.legend()
  plt.savefig("accuracy.png")

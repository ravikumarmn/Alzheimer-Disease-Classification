import matplotlib.pyplot as plt


def plot_graphs(train_losses, test_losses, train_accuracies, test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training loss")
    plt.plot(test_losses, label="Validation loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/train_val_losses.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(test_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("results/train_val_accuracy.png")
    plt.show()
    plt.close()

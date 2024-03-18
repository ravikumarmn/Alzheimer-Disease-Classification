from tqdm import tqdm
from constants import *
import torch
import torch.nn as nn
from model import SimpleCNN
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import json
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import numpy as np

class Trainer:
    def __init__(self, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_losses, self.train_accuracies = [], []
        self.test_losses, self.test_accuracies = [], []
        self.all_labels = 0
        self.all_predictions_probabilities = np.array([])
        self.model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        print("Model training started.")
        tqdm_obj = tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS, leave=False)
        for epoch in tqdm_obj:
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in tqdm(
                self.train_loader, total=len(self.train_loader), leave=False
            ):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            self.train_losses.append(running_loss / len(self.train_loader))
            self.train_accuracies.append(train_accuracy)
            self._evaluate(test_loader=self.test_loader)
            tqdm_obj.set_description_str(
                f"Epoch: {epoch+1}, Train Loss: {self.train_losses[-1]:.4f}, Train Acc: {self.train_accuracies[-1]:.2f}%, Test Loss: {self.test_losses[-1]:.4f}, Test Acc: {self.test_accuracies[-1]:.2f}%"
            )
            
        print("Traning completed.")
    def _evaluate(self, test_loader):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total
        self.test_losses.append(running_loss / len(test_loader))
        self.test_accuracies.append(test_accuracy)

    def evaluate(self):
        self.model.eval()
        running_loss, correct, total = 0, 0, 0
        all_predictions, all_labels = [], []
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                all_predictions.extend(predicted.view(-1).cpu().numpy())
                all_labels.extend(labels.view(-1).cpu().numpy())
        accuracy = 100 * correct / total
        print(f'Test Loss: {running_loss/len(self.test_loader):.4f}, Test Accuracy: {accuracy:.2f}%')
        self.all_labels = all_labels
        return all_labels, all_predictions

    def compute_metrics(self, all_labels, all_predictions):
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro')
        accuracy = accuracy_score(all_labels, all_predictions)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
        with open('results/performance_metric.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        cm = confusion_matrix(all_labels, all_predictions)
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}')
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'], yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig("results/confusion_matrix.png")
        plt.show()

    def plot_roc_curve(self, all_labels, all_predictions_probabilities, num_classes=NUM_CLASSES):
        """Plot and save the ROC curve for each class."""
        # Ensure all_predictions_probabilities is a NumPy array for multidimensional indexing
        all_predictions_probabilities = np.array(all_predictions_probabilities)

        # Binarize the labels for one-vs-all ROC analysis
        all_labels_binarized = label_binarize(all_labels, classes=[*range(num_classes)])
        
        # Compute ROC curve and ROC area for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_binarized[:, i], all_predictions_probabilities[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC')
        plt.legend(loc="lower right")
        plt.savefig("results/roc_curve.png")
        plt.show()

import matplotlib.pyplot as plt

# Local imports
from trainer import Trainer
from data_loader import train_loader, test_loader
from utils import plot_graphs

trainer = Trainer(train_loader=train_loader, test_loader=test_loader)
trainer.train()

plot_graphs(
    trainer.train_losses, 
    trainer.test_losses,
    trainer.train_accuracies, 
    trainer.test_accuracies
)

all_labels, all_predictions = trainer.evaluate()
# trainer.compute_metrics(all_labels, all_predictions)
trainer.plot_roc_curve(all_labels, all_predictions)

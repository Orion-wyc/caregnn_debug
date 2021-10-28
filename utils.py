"""
From GAT utils
"""
import torch
import numpy as np


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def calc_mean_sd(results):
    results = np.around(results, decimals=5)

    MEAN = np.mean(results, axis=0)
    PSD = np.std(results, axis=0)
    SSD = np.std(results, axis=0, ddof=1)
    print(MEAN, PSD, SSD)
    metric_name = ['val_recall', 'val_auc', 'test_recall', 'test_auc']
    for i, name in enumerate(metric_name):
        print("{}= {:1.4f}±{:1.4f}".format(name, MEAN[i], SSD[i]))

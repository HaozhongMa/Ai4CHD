from typing import Any, Dict
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import xgboost as xgb
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)

### Data
dataset = pd.read('data.csv')

task_type = 'multiclass'
assert task_type in ['binclass', 'multiclass', 'regression']


X_all = dataset['data'].astype('float32')
y_all = dataset['target'].astype('float32' if task_type == 'regression' else 'int64')
X_all
if task_type != 'regression':
    y_all = sklearn.preprocessing.LabelEncoder().fit_transform(y_all).astype('int64')

n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None
max(y_all)
X = {}
y = {}
X['train'], X['test'], y['train'], y['test'] = sklearn.model_selection.train_test_split(
    X_all, y_all, train_size=0.8
)
X['train'], X['val'], y['train'], y['val'] = sklearn.model_selection.train_test_split(
    X['train'], y['train'], train_size=0.8
)

# not the best way to preprocess features, but enough for the demonstration
preprocess = sklearn.preprocessing.StandardScaler().fit(X['train'])
X = {
    k: torch.tensor(preprocess.fit_transform(v), device=device)
    for k, v in X.items()
}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}

# !!! CRUCIAL for neural networks when solving regression problems !!!
if task_type == 'regression':
    y_mean = y['train'].mean().item()
    y_std = y['train'].std().item()
    y = {k: (v - y_mean) / y_std for k, v in y.items()}
else:
    y_std = y_mean = None

if task_type != 'multiclass':
    y = {k: v.float() for k, v in y.items()}

### Model
d_out = n_classes or 1

# training
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()


def plot_roc_curves(target, prediction_proba, n_classes, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(target, prediction_proba[:, i], pos_label=i)
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])
    
    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def calculate_metrics(predictions, true_labels, is_regression):
    metrics = {}

    if not is_regression:
        # Calculate probability distribution
        probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()

        # Calculate AUC
        if len(probs.shape) == 2 and probs.shape[1] > 1:  # Multiclass
            metrics['auc'] = roc_auc_score(true_labels, probs, multi_class='ovr')
        else:  # Binary
            metrics['auc'] = roc_auc_score(true_labels, probs[:, 1])

        # Calculate accuracy
        pred_classes = np.argmax(probs, axis=1)
        metrics['accuracy'] = accuracy_score(true_labels, pred_classes)

        # Calculate precision, recall, and F1 scores
        metrics['precision'] = precision_score(true_labels, pred_classes, average='weighted')
        metrics['recall'] = recall_score(true_labels, pred_classes, average='weighted')
        metrics['macro_f1'] = f1_score(true_labels, pred_classes, average='macro')
        metrics['weighted_f1'] = f1_score(true_labels, pred_classes, average='weighted')
        metrics['micro_f1'] = f1_score(true_labels, pred_classes, average='micro')

        # Calculate classification report
        report = classification_report(true_labels, pred_classes, output_dict=True)
        metrics['class_report'] = report

        # Calculate confusion matrix
        metrics['conf_matrix'] = confusion_matrix(true_labels, pred_classes)
        metrics['prediction_proba'] = probs

    return metrics

# Train XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=n_classes, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
    )

xgb_model.fit(X['train'].cpu().numpy(), y['train'].cpu().numpy())

# Predict on validation and test sets
val_preds = xgb_model.predict_proba(X['val'].cpu().numpy())
test_preds = xgb_model.predict_proba(X['test'].cpu().numpy())

# Calculate metrics for validation and test sets
is_regression = False  
val_metrics = calculate_metrics(val_preds, y['val'].cpu().numpy(), is_regression)
test_metrics = calculate_metrics(test_preds, y['test'].cpu().numpy(), is_regression)

# Store best results
best_result = {
    # "epoch": num_epochs, 
    "val": val_metrics,
    "test": test_metrics
}

# Print results
print('*********************************************')
print(f'Best result epoch: {best_result["epoch"]}')
print('---------------------------------------------')
print(f'Best validation auc: {best_result["val"]["auc"]}')
print(f'Best validation accuracy: {best_result["val"]["accuracy"]}')
print(f'Best validation precision: {best_result["val"]["precision"]}')
print(f'Best validation recall: {best_result["val"]["recall"]}')
print(f'Best validation macro f1: {best_result["val"]["macro_f1"]}')
print(f'Best validation weighted f1: {best_result["val"]["weighted_f1"]}')
print(f'Best validation micro f1: {best_result["val"]["micro_f1"]}')
print(f'Best validation class report: {best_result["val"]["class_report"]}')
print('---------------------------------------------')
print(f'Best test auc: {best_result["test"]["auc"]}')
print(f'Best test accuracy: {best_result["test"]["accuracy"]}')
print(f'Best test precision: {best_result["test"]["precision"]}')
print(f'Best test recall: {best_result["test"]["recall"]}')
print(f'Best test macro f1: {best_result["test"]["macro_f1"]}')
print(f'Best test weighted f1: {best_result["test"]["weighted_f1"]}')
print(f'Best test micro f1: {best_result["test"]["micro_f1"]}')
print(f'Best test class report: {best_result["test"]["class_report"]}')

# Write results to file
output_path = 'output_results.txt'
with open(output_path, 'w') as f:
    f.write('*********************************************\\n')
    f.write(f'Best result epoch: {best_result["epoch"]}\\n')
    f.write('---------------------------------------------\\n')
    f.write(f'Best validation auc: {best_result["val"]["auc"]}\\n')
    f.write(f'Best validation accuracy: {best_result["val"]["accuracy"]}\\n')
    f.write(f'Best validation precision: {best_result["val"]["precision"]}\\n')
    f.write(f'Best validation recall: {best_result["val"]["recall"]}\\n')
    f.write(f'Best validation macro f1: {best_result["val"]["macro_f1"]}\\n')
    f.write(f'Best validation weighted f1: {best_result["val"]["weighted_f1"]}\\n')
    f.write(f'Best validation micro f1: {best_result["val"]["micro_f1"]}\\n')
    # f.write(f'Best validation class report: {json.dumps(best_result["val"]["class_report"], indent=4)}\\n')
    f.write('---------------------------------------------\\n')
    f.write(f'Best test auc: {best_result["test"]["auc"]}\\n')
    f.write(f'Best test accuracy: {best_result["test"]["accuracy"]}\\n')
    f.write(f'Best test precision: {best_result["test"]["precision"]}\\n')
    f.write(f'Best test recall: {best_result["test"]["recall"]}\\n')
    f.write(f'Best test macro f1: {best_result["test"]["macro_f1"]}\\n')
    f.write(f'Best test weighted f1: {best_result["test"]["weighted_f1"]}\\n')
    f.write(f'Best test micro f1: {best_result["test"]["micro_f1"]}\\n')
    # f.write(f'Best test class report: {json.dumps(best_result["test"]["class_report"], indent=4)}\\n')

print(f"Results have been written to {output_path}")

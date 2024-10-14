from typing import Any, Dict

import numpy as np
import rtdl
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
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, classification_report, f1_score, accuracy_score
import pandas as pd

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

model = rtdl.FTTransformer.make_default(
    n_num_features=X_all.shape[1],
    cat_cardinalities=None,
    last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
    d_out=d_out,
)

model.to(device)
optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
)

loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
)

# Optimizable-Weights-Loss
# class_count = []
# totel_sample = sum(class_count)
# class_weights = torch.tensor([totel_sample / (len(class_count) * c) for c in class_count],dtype=torch.float,device=device)  
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)


# training

def apply_model(x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )
    
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


@torch.no_grad()
def evaluate(part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        prediction.append(apply_model(batch))
    prediction = torch.cat(prediction).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = scipy.special.expit(prediction)
        binary_prediction = np.round(prediction)
        score = sklearn.metrics.accuracy_score(target, binary_prediction)
        macro_f1 = sklearn.metrics.f1_score(target, binary_prediction, average='macro')
        weighted_f1 = sklearn.metrics.f1_score(target, binary_prediction, average='weighted')
        micro_f1 = sklearn.metrics.f1_score(target, binary_prediction, average='micro')
        conf_matrix = sklearn.metrics.confusion_matrix(target, binary_prediction)
        auc = sklearn.metrics.roc_auc_score(target, prediction)
    elif task_type == 'multiclass':
        prediction_proba = torch.softmax(torch.tensor(prediction), dim=1).numpy()
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
        macro_f1 = sklearn.metrics.f1_score(target, prediction, average='macro')
        weighted_f1 = sklearn.metrics.f1_score(target, prediction, average='weighted')
        micro_f1 = sklearn.metrics.f1_score(target, prediction, average='micro')
        conf_matrix = sklearn.metrics.confusion_matrix(target, prediction)
        num_classes = np.max(target) + 1  # 假设所有的类别都至少出现一次
        class_counts = np.bincount(target, minlength=num_classes)
        class_correct_counts = np.bincount(target[prediction == target], minlength=num_classes)
        class_accuracies = class_correct_counts / class_counts
        auc = sklearn.metrics.roc_auc_score(target, prediction_proba, multi_class='ovr')
        return score, class_accuracies.tolist(), macro_f1, weighted_f1, micro_f1, conf_matrix, auc, prediction_proba
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
        return score
    return score, macro_f1, weighted_f1, micro_f1, conf_matrix, auc

# Create a dataloader for batches of indices
# Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
batch_size = 256
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)


# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=100)


print(f'Test score before training: {evaluate("test")}')
n_epochs = 500
report_frequency = len(X['train']) // batch_size // 5
best_result = {}
for epoch in range(1, n_epochs + 1):
    for iteration, batch_idx in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')
    val_results = evaluate('val')
    test_results = evaluate('test')
    
    val_score, val_accuracies, val_macro_f1, val_weighted_f1, val_micro_f1, val_conf_matrix, val_auc, val_prediction_proba = val_results
    test_score, test_accuracies, test_macro_f1, test_weighted_f1, test_micro_f1, test_conf_matrix, test_auc, test_prediction_proba = test_results

    print(f'Epoch {epoch:03d} \n Validation score: {val_score} \n Test score: {test_score}', end='')
    print(f' Validation Macro F1: {val_macro_f1:.4f} \n Validation Weighted F1: {val_weighted_f1:.4f} \n Validation Micro F1: {val_micro_f1:.4f}')
    print(f' Test Macro F1: {test_macro_f1:.4f} \n Test Weighted F1: {test_weighted_f1:.4f} \n Test Micro F1: {test_micro_f1:.4f}')
    print(f' Validation AUC: {val_auc:.4f} \n Test AUC: {test_auc:.4f}')
    print(f' Validation Confusion Matrix: \n{val_conf_matrix}')
    print(f' Test Confusion Matrix: \n{test_conf_matrix}')

    progress.update(val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
        best_result['epoch'] = epoch
        best_result['val_score'] = val_score
        best_result['test_score'] = test_score
        best_result['test_accuracies'] = val_accuracies
        best_result['test_accuracies'] = test_accuracies
        best_result['val_macro_f1'] = val_macro_f1
        best_result['test_macro_f1'] = test_macro_f1
        best_result['val_weighted_f1'] = val_weighted_f1
        best_result['test_weighted_f1'] = test_weighted_f1
        best_result['val_micro_f1'] = val_micro_f1
        best_result['test_micro_f1'] = test_micro_f1
        best_result['val_conf_matrix'] = val_conf_matrix
        best_result['test_conf_matrix'] = test_conf_matrix
        best_result['val_auc'] = val_auc
        best_result['test_auc'] = test_auc
        best_result['val_prediction_proba'] = val_prediction_proba
        best_result['test_prediction_proba'] = test_prediction_proba

    print()
    if progress.fail:
        break
# plot_confusion_matrix(best_result['val_conf_matrix'], title='Validation Confusion Matrix')
# plot_confusion_matrix(best_result['test_conf_matrix'], title='Test Confusion Matrix')
# plot_roc_curves(y['val'].cpu().numpy(), best_result['val_prediction_proba'], n_classes, title='Validation ROC Curves')
# plot_roc_curves(y['test'].cpu().numpy(), best_result['test_prediction_proba'], n_classes, title='Test ROC Curves')

# best_result
import os
import math
import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from bin import T2GFormer
from lib import Transformations, build_dataset, prepare_tensors, DATA, make_optimizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score,roc_curve,auc


DATASETS = ['test']

IMPLEMENTED_MODELS = [T2GFormer]

AUC_FOR_BINCLASS = False


def get_training_args():
    MODEL_CARDS = [x.__name__ for x in IMPLEMENTED_MODELS]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--model", type=str, default='T2GFormer', choices=MODEL_CARDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=16, help='early stopping for finetune')
    parser.add_argument("--froze_later", action='store_true', help='whether to froze topology in later training phase')
    args = parser.parse_args()

    cfg_file = f'configs/{args.dataset}/{args.model}/cfg.json'
    try:
        print(f"Try to load configuration file: {cfg_file}")
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
    except IOError as e:
        print(f"Not exist !")
        cfg_file = f'configs/default/{args.model}/cfg.json'

        print(f"Try to load default configuration")
        assert os.path.exists(cfg_file), f'Please give a default configuration file: {cfg_file}'
        with open(cfg_file, 'r') as f:
            cfg = json.load(f)
    
    args.output = str(Path(args.output) / f'{args.model}/{args.dataset}/{args.seed}')
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    
    return args, cfg


def record_exp(final_score, best_score, **kwargs):
    results = {
        'config': vars(args),
        'final': final_score,
        'best': best_score,
        **kwargs,
    }
    exp_list = [file for file in os.listdir(args.output) if '.json' in file]
    exp_list = [int(file.split('.')[0]) for file in exp_list]
    exp_id = 0 if len(exp_list) == 0 else max(exp_list) + 1
    with open(f"{args.output}/{exp_id}.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



"""args"""
device = torch.device('cuda')
args, cfg = get_training_args()
seed_everything(args.seed)

"""Datasets and Dataloaders"""
dataset_name = args.dataset
T_cache = True
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / dataset_name, transformation, T_cache)

if AUC_FOR_BINCLASS and dataset.is_binclass:
    metric = 'roc_auc' # AUC (binclass)
else:
    metric = 'score' # RMSE (regression) or ACC (classification)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

if dataset.task_type.value == 'regression':
    y_std = ys['train'].std().item()


batch_size_dict = {
    'test': 128
}
val_batch_size = 1024 if args.dataset in ['santander', 'year', 'microsoft'] else 256 if args.dataset in ['yahoo'] else 8192
if args.dataset == 'epsilon':
    batch_size = 16 if args.dataset == 'epsilon' else 128 if args.dataset == 'yahoo' else 256
elif args.dataset not in batch_size_dict:
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16
else:
    batch_size = batch_size_dict[args.dataset]

num_workers = 0
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

"""Models"""
model_cls = eval(args.model)
n_num_features = dataset.n_num_features
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
cardinalities = None if n_categories == 0 else cardinalities

"""set default"""
cfg['model'].setdefault('kv_compression', None)
cfg['model'].setdefault('kv_compression_sharing', None)
cfg['model'].setdefault('token_bias', True)
# default FR-Graph settings
cfg['model'].setdefault('sym_weight', True)
cfg['model'].setdefault('sym_topology', False)
cfg['model'].setdefault('nsi', True)
"""prepare model arguments"""
kwargs = {
    # task related
    'd_numerical': n_num_features,
    'categories': cardinalities,
    'd_out': d_out,
    **cfg['model']
}
model = model_cls(**kwargs).to(device)

"""Optimizers"""
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

# small learning rate for column embeddings 
# for statble topology learning process
def needs_small_lr(name):
    return any(x in name for x in ['.col_head', '.col_tail'])

for x in ['tokenizer', '.norm', '.bias']:
    assert any(x in a for a in (b[0] for b in model.named_parameters()))
parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k) and not needs_small_lr(k)]
parameters_with_slr = [v for k, v in model.named_parameters() if needs_small_lr(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_with_slr, 'lr': cfg['training']['col_lr'], 'weight_decay': 0.0},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)

# if torch.cuda.device_count() > 1:  # type: ignore[code]
#     print('Using nn.DataParallel')
#     model = nn.DataParallel(model)

"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

# Optimizable-Weights-Loss
# class_count = []
# totel_sample = sum(class_count)
# class_weights = torch.tensor([totel_sample / (len(class_count) * c) for c in class_count],dtype=torch.float,device=device)  
# loss_fn = nn.CrossEntropyLoss(weight=class_weights)


"""utils function"""
def apply_model(x_num, x_cat=None):
    if any(issubclass(eval(args.model), x) for x in IMPLEMENTED_MODELS):
        return model(x_num, x_cat)
    else:
        raise NotImplementedError

@torch.inference_mode()
def evaluate(parts):
    model.eval()
    predictions = {}
    true_labels = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        infer_time = 0.
        predictions[part] = []
        true_labels[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            start = time.time()
            preds = apply_model(x_num, x_cat)
            infer_time += time.time() - start
            predictions[part].append(preds)
            true_labels[part].append(y)
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
        true_labels[part] = torch.cat(true_labels[part]).cpu().numpy()

        if part == 'test':
            print('test time: ', infer_time)
    return predictions, true_labels


def calculate_metrics(predictions, true_labels, is_regression):
    metrics = {}
    
    if not is_regression:
        # 计算概率分布
        probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
        
        # 计算 AUC
        if len(probs.shape) == 2 and probs.shape[1] > 1:  # 多分类
            metrics['auc'] = roc_auc_score(true_labels, probs, multi_class='ovr')
        else:  # 二分类
            metrics['auc'] = roc_auc_score(true_labels, probs[:, 1])
        
        # 计算准确率
        pred_classes = np.argmax(probs, axis=1)
        metrics['accuracy'] = accuracy_score(true_labels, pred_classes)
        
        # 计算 F1 分数
        metrics['macro_f1'] = f1_score(true_labels, pred_classes, average='macro')
        metrics['weighted_f1'] = f1_score(true_labels, pred_classes, average='weighted')
        metrics['micro_f1'] = f1_score(true_labels, pred_classes, average='micro')
        
        # 计算混淆矩阵
        metrics['conf_matrix'] = confusion_matrix(true_labels, pred_classes)
        metrics['prediction_proba'] = probs
    
    else:
        # 回归任务的指标
        raise NotImplementedError("Metrics for regression tasks are not implemented yet.")
    
    return metrics

def record_exp(args, final_score, best_score, **kwargs):
    results = {'config': args, 'final': final_score, 'best': best_score, **kwargs}
    with open(f"{args['output']}/results.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def plot_confusion_matrix(conf_matrix, title,filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filename)


def plot_roc_curves(target, prediction_proba, n_classes, title, filename=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target, prediction_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
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
    plt.savefig(filename)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



"""Training"""
predictions, true_labels = evaluate(['test'])
# we use AUC for binary classification, Accuracy for multi-class classification, RMSE for regression
metric = 'roc_auc' if dataset.is_binclass else 'score'
metrics = calculate_metrics(predictions['test'], true_labels['test'], dataset.is_regression)
print(f'Test metrics: {metrics}')

losses, val_metric, test_metric = [], [], []
n_epochs = 10000
report_frequency = len(ys['train']) // batch_size // 3
loss_holder = AverageMeter()
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
no_improvement = 0
frozen_switch = args.froze_later # whether to froze topology in later training phase
EARLY_STOP = args.early_stop
best_result={}

for epoch in range(1, n_epochs + 1):
    model.train()
    for iteration, batch in enumerate(train_loader):
        x_num, x_cat, y = (
            (batch[0], None, batch[1])
            if len(batch) == 2
            else batch
        )
        optimizer.zero_grad()
        loss = loss_fn(apply_model(x_num, x_cat), y)
        loss.backward()
        optimizer.step()
        loss_holder.update(loss.item(), len(ys))
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss_holder.val:.4f} (avg_loss) {loss_holder.avg:.4f}')
    losses.append(loss_holder.avg)
    loss_holder.reset()
    
    # 计算指标
    predictions, true_labels = evaluate(['train', 'val', 'test'])
    val_metrics = calculate_metrics(predictions['val'], true_labels['val'], dataset.is_regression)
    test_metrics = calculate_metrics(predictions['test'], true_labels['test'], dataset.is_regression)
    
    # 使用正确的指标名称
    val_score = val_metrics.get('accuracy', -np.inf)  
    test_score = test_metrics.get('accuracy', -np.inf) 

    if val_score > best_score:
        best_score = val_score
        final_test_score = test_score
        print(' <<< BEST VALIDATION EPOCH')
        best_result['epoch'] = epoch
        best_result['val'] = val_metrics
        best_result['test'] = test_metrics
        no_improvement = 0
    else:
        no_improvement += 1
    if test_score > best_test_score:
        best_test_score = test_score

    if no_improvement == EARLY_STOP:
        break


print(f'Best result epoch: {best_result["epoch"]}')
print(f'Best validation auc: {best_result["val"]["auc"]}')
print(f'Best test auc: {best_result["test"]["auc"]}')
print(f'Best validation accuracy: {best_result["val"]["accuracy"]}')
print(f'Best test accuracy: {best_result["test"]["accuracy"]}')
print(f'Best validation macro f1: {best_result["val"]["macro_f1"]}')
print(f'Best test macro f1: {best_result["test"]["macro_f1"]}')
print(f'Best validation weighted f1: {best_result["val"]["weighted_f1"]}')
print(f'Best test weighted f1: {best_result["test"]["weighted_f1"]}')
print(f'Best validation micro f1: {best_result["val"]["micro_f1"]}')
print(f'Best test micro f1: {best_result["test"]["micro_f1"]}')




plot_confusion_matrix(best_result['val']['conf_matrix'], title='Validation Confusion Matrix',filename=f"{args.output}/val_confusion_matrix.png")
plot_confusion_matrix(best_result['test']['conf_matrix'], title='Test Confusion Matrix',filename=f"{args.output}/test_confusion_matrix.png")

plot_roc_curves(true_labels['val'], best_result['val']['prediction_proba'], 3, title='Validation ROC Curves', filename=f"{args.output}/val_roc_curves.png")
plot_roc_curves(true_labels['test'], best_result['test']['prediction_proba'], 3, title='Test ROC Curves', filename=f"{args.output}/test_roc_curves.png")


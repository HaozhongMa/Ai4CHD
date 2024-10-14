import os
import math
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score,roc_curve,auc
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, make_optimizer, DATA


DATASETS = ['test']

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='result/ExcelFormer/default')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.5, help='hyper-parameter of Beta Distribution in mixup, we choose 0.5 for all datasets in default config')
    parser.add_argument("--mix_type", type=str, default='none', choices=['niave_mix', 'feat_mix', 'hidden_mix', 'none'], help='mixup type, set to "niave_mix" for naive mixup, set to "none" if no mixup')
    parser.add_argument("--save", action='store_true', help='whether to save model')
    parser.add_argument("--catenc", action='store_true', help='whether to use catboost encoder for categorical features')
    args = parser.parse_args()

    args.output = f'{args.output}/mixup({args.mix_type})/{args.dataset}/{args.seed}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    # some basic model configuration
    cfg = {
        "model": {
            "prenormalization": True, # true or false, perform BETTER on a few datasets with no prenormalization 

            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }
    
    return args, cfg

def record_exp(args, final_score, best_score, **kwargs):
    # 'best': the best test score during running
    # 'final': the final test score acquired by validation set
    results = {'config': args, 'final': final_score, 'best': best_score, **kwargs}
    with open(f"{args['output']}/results.json", 'w', encoding='utf8') as f:
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

""" prepare Datasets and Dataloaders """
assert args.dataset in DATASETS
T_cache = False # save data preprocessing cache
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)
dataset = build_dataset(DATA / args.dataset, transformation, T_cache)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
# convert categorical features to numerical features with CatBoostEncoder
if args.catenc and dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    for k in ['train', 'val', 'test']:
        # 1: directly regard catgorical features as numerical
        dataset.X_num[k] = np.concatenate([enc.transform(dataset.X_cat[k]).astype(np.float32), dataset.X_num[k]], axis=1)

d_out = dataset.n_classes or 1
X_num, X_cat, ys = prepare_tensors(dataset, device=device)

if args.catenc: # if use CatBoostEncoder then drop original categorical features
    X_cat = None

""" ORDER numerical features with MUTUAL INFORMATION """
mi_cache_dir = 'cache/mi'
if not os.path.isdir(mi_cache_dir):
    os.makedirs(mi_cache_dir)
mi_cache_file = f'{mi_cache_dir}/{args.dataset}.npy' # cache to save mutual information
if os.path.exists(mi_cache_file):
    mi_scores = np.load(mi_cache_file)
else:
    mi_func = mutual_info_regression if dataset.is_regression else mutual_info_classif
    mi_scores = mi_func(dataset.X_num['train'], dataset.y['train']) # calculate MI
    np.save(mi_cache_file, mi_scores)
mi_ranks = np.argsort(-mi_scores)
# reorder the feature with mutual information ranks
X_num = {k: v[:, mi_ranks] for k, v in X_num.items()}
# normalized mutual information for loss weight
sorted_mi_scores = torch.from_numpy(mi_scores[mi_ranks] / mi_scores.sum()).float().to(device)
""" END FEATURE REORDER """

# set batch size
batch_size_dict = {
    'test': 128,
} 

if args.dataset in batch_size_dict:
    batch_size = batch_size_dict[args.dataset]
    val_batch_size = 512
else:
    # batch size settings for datasets in (Grinsztajn et al., 2022)
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

# update training config
cfg['training'].update({
    "batch_size": batch_size, 
    "eval_batch_size": val_batch_size, 
    "patience": args.early_stop
})

# data loaders
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

""" Prepare Model """
# datset specific params
n_num_features = dataset.n_num_features # drop some features
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
if args.catenc:
    n_categories = 0 # all categorical features are converted to numerical ones
cardinalities = None if n_categories == 0 else cardinalities # drop category features

""" All default configs: model and training hyper-parameters """
# kwargs: model configs
kwargs = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model']
}
default_model_configs = {
    'ffn_dropout': 0., 'attention_dropout': 0.3, 'residual_dropout': 0.0,
    'n_layers': 3, 'n_heads': 32, 'd_token': 256,
    'init_scale': 0.01, # param for the Attenuated Initialization
}
default_training_configs = {
    'lr': 1e-4,
    'weight_decay': 0.,
}
kwargs.update(default_model_configs) # update model configs
cfg['training'].update(default_training_configs) # update training configs

# build model
model = ExcelFormer(**kwargs).to(device)

# optimizer
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)

# parallelization
# if torch.cuda.device_count() > 1:
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



"""Utils Function"""
def apply_model(x_num, x_cat=None, mixup=False):
    if mixup:
        return model(x_num, x_cat, mixup=True, beta=args.beta, mtype=args.mix_type)
    return model(x_num, x_cat)

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
        
        probs = torch.softmax(torch.from_numpy(predictions), dim=1).numpy()
        if len(probs.shape) == 2 and probs.shape[1] > 1:  
            metrics['auc'] = roc_auc_score(true_labels, probs, multi_class='ovr')
        else:  
            metrics['auc'] = roc_auc_score(true_labels, probs[:, 1])
        pred_classes = np.argmax(probs, axis=1)
        metrics['accuracy'] = accuracy_score(true_labels, pred_classes)

        metrics['macro_f1'] = f1_score(true_labels, pred_classes, average='macro')
        metrics['weighted_f1'] = f1_score(true_labels, pred_classes, average='weighted')
        metrics['micro_f1'] = f1_score(true_labels, pred_classes, average='micro')

        metrics['conf_matrix'] = confusion_matrix(true_labels, pred_classes)
        metrics['prediction_proba'] = probs
    
    else:

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
n_epochs = 500 # default max training epoch

# warmup and lr scheduler
warm_up = 10 # warm up epoch
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs - warm_up) # lr decay
max_lr = cfg['training']['lr']
report_frequency = len(ys['train']) // batch_size // 3

# metric containers
loss_holder = AverageMeter()
best_score = -np.inf
final_test_score = -np.inf # final test score acquired by max validation set score
best_test_score = -np.inf # best test score during running
running_time = 0.

# early stop
no_improvement = 0
EARLY_STOP = args.early_stop
best_result = {}

for epoch in range(1, n_epochs + 1):
    model.train()
    if warm_up > 0 and epoch <= warm_up:
        lr = max_lr * epoch / warm_up
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
    
    for iteration, batch in enumerate(train_loader):
        x_num, x_cat, y = (
            (batch[0], None, batch[1])
            if len(batch) == 2
            else batch
        )
        
        start = time.time()
        optimizer.zero_grad()
        if args.mix_type == 'none':
            loss = loss_fn(apply_model(x_num, x_cat, mixup=False), y)
        else:
            preds, feat_masks, shuffled_ids = apply_model(x_num, x_cat, mixup=True)
            if args.mix_type == 'feat_mix':
                lambdas = (sorted_mi_scores * feat_masks).sum(1)
                lambdas2 = 1 - lambdas
            elif args.mix_type == 'hidden_mix':
                lambdas = feat_masks
                lambdas2 = 1 - lambdas
            elif args.mix_type == 'niave_mix':
                lambdas = feat_masks
                lambdas2 = 1 - lambdas
            if dataset.is_regression:
                mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                loss = loss_fn(preds, mix_y)
            else:
                loss = lambdas * loss_fn(preds, y, reduction='none') + lambdas2 * loss_fn(preds, y[shuffled_ids], reduction='none')
                loss = loss.mean()
        loss.backward()
        optimizer.step()
        running_time += time.time() - start
        loss_holder.update(loss.item(), len(ys))
        if iteration % report_frequency == 0:
            print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss_holder.val:.4f} (avg_loss) {loss_holder.avg:.4f}')
    losses.append(loss_holder.avg)
    loss_holder.reset()

    predictions, true_labels = evaluate(['train', 'val', 'test'])
    val_metrics = calculate_metrics(predictions['val'], true_labels['val'], dataset.is_regression)
    test_metrics = calculate_metrics(predictions['test'], true_labels['test'], dataset.is_regression)
    
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
        if args.save:
            torch.save(model.state_dict(), f"{args.output}/pytorch_model.pt")
    else:
        no_improvement += 1
    if test_score > best_test_score:
        best_test_score = test_score

    if no_improvement == EARLY_STOP:
        break

print(best_result)

# plot_confusion_matrix(best_result['val']['conf_matrix'], title='Validation Confusion Matrix',filename=f"{args.output}/val_confusion_matrix.png")
# plot_confusion_matrix(best_result['test']['conf_matrix'], title='Test Confusion Matrix',filename=f"{args.output}/test_confusion_matrix.png")

# plot_roc_curves(true_labels['val'], best_result['val']['prediction_proba'], 3, title='Validation ROC Curves', filename=f"{args.output}/val_roc_curves.png")
# plot_roc_curves(true_labels['test'], best_result['test']['prediction_proba'], 3, title='Test ROC Curves', filename=f"{args.output}/test_roc_curves.png")

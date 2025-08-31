''' Evaluation metrics
'''

from torch import Tensor
import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# -- auroc
def mean_auroc(
    scores,
    y_true,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
):
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().detach().numpy()
    
    if isinstance(scores, Tensor):
        scores = scores.cpu().detach().numpy()
    
    roc_scores = []
    
    for c in np.unique(y_true):
        if c == 0:
            # no auroc for benign data
            continue
        
        # get attack class and benign traffic
        class_mask = (y_true == 0) | (y_true == c)
        y = y_true[class_mask]  
        y[y>0] = 1

        x = scores[class_mask]
        roc = roc_auc_score(y,x)
        if include_lower_thres:
            roc_scores.append(max(roc, 1-roc))
        else:
            roc_scores.append(roc) 
    
    if return_class_level:
        return roc_scores
    else:
        return roc_scores[-1]

def balanced_auroc(scores, labels, return_class_level: bool = False):
    class_auroc = mean_auroc(scores = scores, y_true = labels, return_class_level= True)
    
    if return_class_level:
        return class_auroc
    else:
        return class_auroc[:-1]

# -- supervised metrics
def evaluate_metrics(y_true, y_pred, prefix = ''):
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}mean_recall": recall_score(y_true, y_pred, average="macro"),
        f"{prefix}mean_precision": precision_score(y_true, y_pred, average="macro"),
        f"{prefix}macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics
''' Evaluation metrics
'''

from torch import Tensor
import numpy as np
from numpy import ndarray
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

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

# -- macro f1 score
def macro_f1_score(y_true: ndarray, y_pred: ndarray) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division = 0.0)

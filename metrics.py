import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,roc_auc_score,matthews_corrcoef
from sklearn.metrics import precision_recall_curve,average_precision_score
from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error
from scipy import stats
import numpy as np 

def compute_cls_metrics(y_true, y_prob):
    y_pred = np.array(y_prob) > 0.5
    roc_auc = roc_auc_score(y_true, y_prob)
    F1 = f1_score(y_true, y_pred, average = 'binary')
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return   F1, roc_auc, mcc,  tn, fp, fn, tp


def compute_reg_metrics(y_true, y_prob):
    y_true = y_true.flatten().astype(float)
    y_prob = y_prob.flatten().astype(float)
    tau, p_value = stats.kendalltau(y_true, y_prob)
    rho, pval =stats.spearmanr(y_true, y_prob)
    r, _ =stats.pearsonr(y_true, y_prob)
    rmse = mean_squared_error(y_true, y_prob, squared=False)
    mae = mean_absolute_error(y_true, y_prob)
    return tau, rho, r, rmse, mae
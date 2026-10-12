import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, roc_curve, precision_recall_curve
)
from .utils import save_figure, OUTPUTS_FIGURES_DIR, OUTPUTS_TABLES_DIR

def evaluate_classification(model, X, y, threshold=0.5, model_name="Model"):
    """
    Computes a comprehensive suite of academic evaluation metrics.
    Essential for imbalanced sports data where accuracy alone is deceptive.
    """
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        df = model.decision_function(X)
        y_proba = (df - df.min()) / (df.max() - df.min() + 1e-8)
    else:
        y_proba = model.predict(X)
        
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y, y_proba)
    except Exception:
        roc_auc = np.nan
        
    try:
        pr_auc = average_precision_score(y, y_proba)
    except Exception:
        pr_auc = np.nan
        
    brier = brier_score_loss(y, y_proba)
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(acc, 4),
        'Balanced_Accuracy': round(bal_acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'Specificity': round(spec, 4),
        'F1_Score': round(f1, 4),
        'ROC_AUC': round(roc_auc, 4),
        'PR_AUC': round(pr_auc, 4),
        'Brier_Score': round(brier, 4),
        'Threshold': threshold
    }
    
    return metrics, cm, y_proba

def plot_confusion_matrix(cm, model_name="Model", filename=None):
    """Plots count and normalized confusion matrix side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Standard', 'High Performer'],
                yticklabels=['Standard', 'High Performer'])
    axes[0].set_title(f"{model_name} - Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
                xticklabels=['Standard', 'High Performer'],
                yticklabels=['Standard', 'High Performer'])
    axes[1].set_title(f"{model_name} - Normalized")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig

def plot_roc_and_pr_curves(y_true, y_proba, model_name="Model", filename=None):
    """Plots ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Chance (AUC = 0.500)')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].set_title(f'ROC Curve - {model_name}')
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)
    
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline_rate = y_true.sum() / len(y_true)
    axes[1].plot(rec, prec, color='teal', lw=2, label=f'PR curve (AP = {pr_auc:.3f})')
    axes[1].plot([0, 1], [baseline_rate, baseline_rate], color='grey', lw=1.5, linestyle='--',
                 label=f'Baseline Rate ({baseline_rate:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve - {model_name}')
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    if filename:
        save_figure(fig, filename)
    return fig

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Finds the probability threshold that maximizes F1-score."""
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = []
    for t in thresholds:
        y_p = (y_proba >= t).astype(int)
        if metric == 'f1':
            scores.append(f1_score(y_true, y_p, zero_division=0))
        elif metric == 'balanced_accuracy':
            scores.append(balanced_accuracy_score(y_true, y_p))
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]

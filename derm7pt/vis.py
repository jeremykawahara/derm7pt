import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, labels, fontsize=18,
                   figsize=(16, 12), cmap=plt.cm.coolwarm_r, ax=None, colorbar=True,
                   xrotation=30, yrotation=30):

    label_indexes = np.arange(0, len(labels))

    cm = confusion_matrix(y_true, y_pred, label_indexes)
    # Normalized per-class.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        # Create a new figure if no axis is specified.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    cax = ax.matshow(cm_normalized, vmin=0, vmax=1, alpha=0.8, cmap=cmap)

    # Print the number of samples that fall within each cell.
    x, y = np.meshgrid(label_indexes, label_indexes)
    for (x_val, y_val) in zip(x.flatten(), y.flatten()):
        c = cm[int(x_val), int(y_val)]
        ax.text(y_val, x_val, c, va='center', ha='center', fontsize=fontsize)

    if colorbar:
        cb = plt.colorbar(cax, fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=fontsize)

    # Make the confusion matrix pretty.
    ax.xaxis.tick_bottom()
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    plt.xticks(label_indexes, rotation=xrotation)
    plt.yticks(label_indexes, rotation=yrotation, va='center', x=0.05)
    plt.ylabel('True label', fontweight='bold', fontsize=fontsize)
    plt.xlabel('Predicted label', fontweight='bold', fontsize=fontsize)
    ax.set_xticks(np.arange(-.5, len(ax.get_xticks()), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(ax.get_yticks()), 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.grid(b=False, which='major')

    return cax

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def renderPointsWithDecisionBounds(XXX, yyy, score, clf,
                                   h=.02,  # step size in the mesh
                                   cm=plt.cm.RdBu,
                                   plot_margin=.2
                                   ):
    fig = plt.figure(figsize=(12, 7))
    ax = plt.subplot(111)

    x_min, x_max = XXX[:, 0].min() - plot_margin, XXX[:, 0].max() + plot_margin
    y_min, y_max = XXX[:, 1].min() - plot_margin, XXX[:, 1].max() + plot_margin
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    scatter_2d_label(XXX, y=yyy, alpha=0.2, s=1, lw=1, ax=ax)

    #     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright) # Plot also the training points
    #     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6) # and testing points

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    #     if ds_cnt == 0:
    #         ax.set_title(name)

    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=15, horizontalalignment='right')


def scatter_2d_label(X_2d, y, s=2, alpha=0.5, lw=2, ax=plt):
    """Visualuse a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.

    s : float
        Marker size for scatter plot.

    alpha : float
        Transparency for scatter plot.

    lw : float
        Linewidth for scatter plot.
    """
    targets = np.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        ax.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)
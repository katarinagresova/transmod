import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def scatterplot_with_density(x, y, xlabel='X', ylabel='Y', title='Scatterplot with Density'):

    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)(values)

    plt.figure(figsize=(6,6))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    min_lim = min(np.min(x), np.min(y))
    max_lim = max(np.max(x), np.max(y))
    xlim = (min_lim - 0.05 * abs(min_lim), max_lim + 0.05 * abs(max_lim))
    ylim = (min_lim - 0.05 * abs(min_lim), max_lim + 0.05 * abs(max_lim))

    plt.xlim(xlim)
    plt.ylim(ylim)

    sns.scatterplot(x=x, y=y, alpha=1, c=kernel, cmap='viridis', edgecolor=None, s=20)
    plt.plot([min_lim - 1, max_lim + 1], [min_lim - 1, max_lim + 1], color='red', linestyle='--')
    plt.show()
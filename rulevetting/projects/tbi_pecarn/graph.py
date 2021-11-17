import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Graph functions for EDA & modeling.
'''

def barplot(series, savefig, title='', xlab=''):
    """sns barplot of pandas series
    :param: series - pandas series
    :param: savefig - boolean - save fig or not
    :param: title, xlab - strings for plot
    :return: plt fig
    """
    plt.figure(num=None, figsize=(20,18), dpi=80, facecolor='w', edgecolor='r')
    fig = sns.barplot(series.values, series.index)
    plt.suptitle(title)
    plt.xlabel(xlab)
    if savefig:
        plt.savefig(f'figs/{title}.png', dpi=350)
    
    return fig


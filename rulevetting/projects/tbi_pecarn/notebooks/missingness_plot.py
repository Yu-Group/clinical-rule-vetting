# Data utilities
import pandas as pd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting font sizes and properties
TINY_SIZE = 12
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 20
MARKER_SIZE = 6
LINE_SIZE = 4

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=TINY_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("lines", markersize=MARKER_SIZE)  # marker size
plt.rc("lines", linewidth=LINE_SIZE)  # line width

mpl.rcParams["figure.dpi"] = 180

# Height and width per row and column of subplots
FIG_HEIGHT = 20
FIG_WIDTH = 18

fig_fcn = lambda kwargs: plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), **kwargs)
color_list = sns.color_palette("colorblind")


def barplot(
    series,  # type: pd.Series
    savefig=False,  # type: bool
    title="",  # type: str
    xlab="",  # type: str
    fig_path="figs/",  # type: str
):  # type (...) -> plt.figure
    """
        Seaborn barplot of Pandas series

    :param: series - pandas series
    :param: savefig - boolean - save fig or not
    :param: title, xlab - strings for plot
    :param: fig_path - string for figure folder/path
    :return: matplotlib figure
    """

    fig_fcn({"num": None, "dpi": 80, "facecolor": "w", "edgecolor": "r"})
    fig = sns.barplot(series.values, series.index, palette="colorblind")
    plt.title(title)
    plt.xlabel(xlab)
    if savefig:
        plt.savefig(fig_path + f"{title}.png", dpi=350)

    return fig


# Load data
filename = "./rule-vetting-master/data/tbi_pecarn/raw/TBI PUD 10-08-2013.csv"
data = pd.read_csv(filename)

# Make plot
missingness = (data == 92).mean() # pd.isnull(data).mean()
missingness = missingness[missingness > 0]
missingness = missingness.sort_values(ascending=False)
barplot(
    missingness, False, "Invalidness by Variable, TBI Pecarn", "Fraction of Invalid Samples"
)

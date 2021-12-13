import seaborn as sns
import matplotlib.pyplot as plt

cmap_div = sns.diverging_palette(10, 220, as_cmap=True)
cb2 = '#66ccff'
cb = '#1f77b4'
cr = '#cc0000'
cp = '#cc3399'
cy = '#d8b365'
cg = '#5ab4ac'
cm = sns.diverging_palette(10, 240, n=1000, as_cmap=True)
cm_rev = sns.diverging_palette(240, 10, n=1000, as_cmap=True)


def jointplot_grouped(col_x: str, col_y: str, col_k: str, df,
                      k_is_color=False, scatter_alpha=.5, add_global_hists: bool = True):
    '''Jointplot of hists + densities
    Params
    ------
    col_x
        name of X var
    col_y
        name of Y var
    col_k
        name of variable to group/color by
    add_global_hists
        whether to plot the global hist as well
    '''

    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    color = None
    legends = []
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color = name
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,
            vertical=True
        )
    if add_global_hists:
        sns.distplot(
            df[col_x].values,
            ax=g.ax_marg_x,
            color='grey'
        )
        sns.distplot(
            df[col_y].values.ravel(),
            ax=g.ax_marg_y,
            color='grey',
            vertical=True
        )
    plt.legend(legends)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _get_data(sens_level, metric):
    f_name = f"results/tbi_{metric}_{sens_level}.csv"
    df =  pd.read_csv(f_name, index_col=0)
    df.columns = [f"V {i}" for i in range(df.shape[1])]
    df.index = [f"Model {i}" for i in range(df.shape[0])]
    return df


def test_set_stability(metric, sens_lev):
    data_spec = _get_data(sens_lev, metric)
    sns.boxplot(x="variable", y="value", data=pd.melt(data_spec))
    label_dict = {"sens":"Sensitivity", "spec":"Specificity"}
    plt.ylabel(label_dict[metric])
    plt.xlabel("Data Version")
    # plt.title(f"{label_dict[metric]} variation across RuleFit models, for each data version")
    plt.savefig(f"results/ds_stability_{sens_lev}_{metric}.png", dpi=300)
    plt.close()


def model_stability(metric, sens_lev):
    data_spec = _get_data(sens_lev, metric)
    sns.boxplot(x="variable", y="value", data=pd.melt(data_spec.T))
    label_dict = {"sens": "Sensitivity", "spec": "Specificity"}
    plt.ylabel(label_dict[metric])
    plt.xlabel("RuleFit Version")
    # plt.title(f"{label_dict[metric]} variation across RuleFit models, for each data version")
    plt.savefig(f"results/model_{sens_lev}_{metric}.png", dpi=300)
    plt.close()


def main():
    model_stability(sens_lev=95, metric="spec")
    model_stability(sens_lev=95, metric="sens")

    test_set_stability(sens_lev=95, metric="spec")
    test_set_stability(sens_lev=95, metric="sens")


if __name__ == '__main__':
    main()

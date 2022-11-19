import os
import random
import sys
import warnings
import webbrowser
from typing import List

import numpy as np
import pandas as pd
import seaborn
from plotly import express as px
from plotly import graph_objects as go
from scipy import stats
from sklearn import datasets
from sklearn.metrics import confusion_matrix

rootpath = os.path.dirname(sys.argv[0])
urlpath = "%s/midtermplots/" % (rootpath)

TITANIC_PREDICTORS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "embarked",
    "parch",
    "fare",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alone",
    "class",
]


def get_test_data_set(data_set_name: str = None) -> (pd.DataFrame, List[str], str):
    """Function to load a few test data sets
    :param:
    data_set_name : string, optional
        Data set to load

    :return:
    data_set : :class:`pandas.DataFrame`
        Tabular data, possibly with some preprocessing applied.
    predictors :list[str]
        List of predictor variables
    response: str
        Response variable
    """
    seaborn_data_sets = ["mpg", "tips", "titanic", "titanic_2"]
    sklearn_data_sets = ["boston", "diabetes", "breast_cancer"]
    all_data_sets = seaborn_data_sets + sklearn_data_sets

    if data_set_name is None:
        data_set_name = random.choice(all_data_sets)
    else:
        if data_set_name not in all_data_sets:
            raise Exception(f"Data set choice not valid: {data_set_name}")

    if data_set_name in seaborn_data_sets:
        if data_set_name == "mpg":
            data_set = seaborn.load_dataset(name="mpg").dropna().reset_index()
            predictors = [
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "origin",
                "name",
            ]
            response = "mpg"
        elif data_set_name == "tips":
            data_set = seaborn.load_dataset(name="tips").dropna().reset_index()
            predictors = [
                "total_bill",
                "sex",
                "smoker",
                "day",
                "time",
                "size",
            ]
            response = "tip"
        elif data_set_name == "titanic":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "survived"
        elif data_set_name == "titanic_2":
            data_set = seaborn.load_dataset(name="titanic").dropna()
            predictors = TITANIC_PREDICTORS
            response = "alive"
    elif data_set_name in sklearn_data_sets:
        if data_set_name == "boston":
            data = datasets.load_boston()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["CHAS"] = data_set["CHAS"].astype(str)
        elif data_set_name == "diabetes":
            data = datasets.load_diabetes()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)
            data_set["gender"] = ["1" if i > 0 else "0" for i in data_set["sex"]]
        elif data_set_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            data_set = pd.DataFrame(data.data, columns=data.feature_names)

        data_set["target"] = data.target
        predictors = data.feature_names
        response = "target"

    print(f"Data set selected: {data_set_name}")
    return data_set, predictors, response


def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cont_cont_correlation(x, y):
    """
    Calculates correlation statistic for continuous-continuous association.
    Pearson Correlation Coefficient
    https://medium.com/geekculture/how-to-find-the-correlation-between-continuous-variables-and-visualise-it-using-python-7faf5b028ae0
    """
    r, p_value = stats.pearsonr(x, y)
    return (r, p_value)


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        x, y = fill_na(x), fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_observations = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_observations

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_corrected = max(0, phi2 - ((r - 1) * (c - 1)) / (n_observations - 1))
            r_corrected = r - ((r - 1) ** 2) / (n_observations - 1)
            c_corrected = c - ((c - 1) ** 2) / (n_observations - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_corrected / np.sqrt((r_corrected - 1) * (c_corrected - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_corrected / min((r_corrected - 1), (c_corrected - 1))
            )
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cont_correlation_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2))
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def checkConsecutive(numlist):
    return sorted(numlist) == list(range(min(numlist), max(numlist) + 1))


def determin_type(df, col):
    col_type = df[col].dtype
    if col_type == "float64":
        return "cont"
    elif col_type == "int64":
        # check distinct values
        distinct = list(set(df[col].tolist()))
        if len(distinct) / len(df) < 0.005:
            return "cat"
        else:
            if checkConsecutive(distinct):
                return "cat"
            else:
                return "cont"
    else:
        return "cat"


# Continuous/Continuous Predictor Pairs
def cal_cont_cont_corre(pred_cont_df):
    cont_cont_list = []
    for index in range(pred_cont_df.shape[1]):
        index2 = index + 1
        if index2 < pred_cont_df.shape[1]:
            for index2 in range(index2, pred_cont_df.shape[1]):
                cont_cont_dic = {}
                pair_name = (
                    pred_cont_df.columns[index] + " and " + pred_cont_df.columns[index2]
                )
                x = pred_cont_df.iloc[:, index]
                y = pred_cont_df.iloc[:, index2]
                r, p_value = cont_cont_correlation(x, y)
                cont_cont_dic["Predictors"] = pair_name
                cont_cont_dic["Pearson's r"] = r
                cont_cont_dic["Absolute Value of Correlation"] = abs(r)

                t_value, p = stats.ttest_rel(x, y)

                # Plot the figure
                fig = px.scatter(x=x, y=y, trendline="ols")
                fig.update_layout(
                    title=f"Variable: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=pred_cont_df.columns[index],
                    yaxis_title=pred_cont_df.columns[index2],
                )
                # fig.show()
                fig.write_html(
                    file="%s/midtermplots/%s_linear_regression.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cont_cont_dic["Linear Regression Plot"] = (
                    pair_name.replace(" ", "_") + "_linear_regression"
                )
                cont_cont_list.append(cont_cont_dic)

    cont_cont_list = sorted(
        cont_cont_list, key=lambda x: x["Absolute Value of Correlation"], reverse=True
    )
    cont_cont_df = pd.DataFrame.from_dict(cont_cont_list)
    # create html link
    if "Linear Regression Plot" in cont_cont_df.columns:
        cont_cont_df["Linear Regression Plot"] = cont_cont_df[
            "Linear Regression Plot"
        ].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return cont_cont_df


# Continuous/Categorical Predictor Pairs
def cal_cont_cat_corre(pred_cont_df, pred_cat_df, df):
    cont_cat_list = []
    for c1 in pred_cont_df.columns:
        for c2 in pred_cat_df.columns:
            cont_cat_dic = {}
            pair_name = c1 + " and " + c2
            corr = cat_cont_correlation_ratio(
                pred_cat_df[c2].values, pred_cont_df[c1].values
            )
            cont_cat_dic["Predictors"] = pair_name
            cont_cat_dic["Correlation Ratio"] = corr
            cont_cat_dic["Absolute Value of Correlation"] = abs(corr)

            # violin plot
            group_labels = list(set(pred_cat_df[c2].values))
            fig_violin = go.Figure()
            for curr_group in group_labels:
                fig_violin.add_trace(
                    go.Violin(
                        x=df[c2][df[c2] == curr_group],
                        y=df[c1][df[c2] == curr_group],
                        name=str(curr_group),
                        legendgrouptitle_text=c2,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

            fig_violin.update_layout(
                title="%s by %s" % (c2, c1), xaxis_title=c2, yaxis_title=c1
            )
            # fig_violin.show()
            fig_violin.write_html(
                file="%s/midtermplots/%s_violin_plot.html"
                % (rootpath, pair_name.replace(" ", "_")),
                include_plotlyjs="cdn",
            )

            cont_cat_dic["Violin Plot"] = pair_name.replace(" ", "_") + "_violin_plot"

            # dist plot
            fig_dist = px.histogram(
                df, x=c1, color=c2, marginal="rug", hover_data=df.columns
            )
            fig_dist.update_layout(
                title="%s by %s" % (c1, c2), xaxis_title=c2, yaxis_title=c1
            )
            # fig_dist.show()
            fig_dist.write_html(
                file="%s/midtermplots/%s_dist_plot.html"
                % (rootpath, pair_name.replace(" ", "_")),
                include_plotlyjs="cdn",
            )

            cont_cat_dic["Distribution Plot"] = (
                pair_name.replace(" ", "_") + "_dist_plot"
            )
            cont_cat_list.append(cont_cat_dic)

    cont_cat_list = sorted(
        cont_cat_list, key=lambda x: x["Absolute Value of Correlation"], reverse=True
    )
    con_cat_df = pd.DataFrame.from_dict(cont_cat_list)

    if "Violin Plot" in con_cat_df.columns:
        con_cat_df["Violin Plot"] = con_cat_df["Violin Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )
    if "Distribution Plot" in con_cat_df.columns:
        con_cat_df["Distribution Plot"] = con_cat_df["Distribution Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return con_cat_df


# Categorical/Categorical Predictor Pairs
def cal_cat_cat_corre(pred_cat_df):
    cat_cat_list = []
    for index in range(0, pred_cat_df.shape[1]):
        index2 = index + 1
        if index2 < pred_cat_df.shape[1]:
            for index2 in range(index2, pred_cat_df.shape[1]):
                c1 = pred_cat_df.columns[index]
                c2 = pred_cat_df.columns[index2]
                cat_cat_dic = {}
                pair_name = c1 + " and " + c2
                corr = cat_correlation(pred_cat_df[c1], pred_cat_df[c2])
                cat_cat_dic["Predictors"] = pair_name
                cat_cat_dic["Cramer's V"] = corr
                cat_cat_dic["Absolute Value of Correlation"] = abs(corr)

                conf_matrix = confusion_matrix(
                    pred_cat_df[c1].astype(str), pred_cat_df[c2].astype(str)
                )
                fig_cat = go.Figure(
                    data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
                )
                fig_cat.update_layout(
                    title="%s %s Heatmap" % (c1, c2), xaxis_title=c1, yaxis_title=c2
                )
                # fig_cat.show()
                fig_cat.write_html(
                    file="%s/midtermplots/%s_heatmap_plot.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cat_cat_dic["Heatmap"] = pair_name.replace(" ", "_") + "_heatmap_plot"
                cat_cat_list.append(cat_cat_dic)

    cat_cat_list = sorted(
        cat_cat_list, key=lambda x: x["Absolute Value of Correlation"], reverse=True
    )
    cat_cat_df = pd.DataFrame.from_dict(cat_cat_list)
    if "Heatmap" in cat_cat_df.columns:
        cat_cat_df["Heatmap"] = cat_cat_df["Heatmap"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return cat_cat_df


def cal_cont_cont_mwr(df, pred_cont_col, response_col):
    cont_cont_mwr_list = []
    col = pred_cont_col[:]
    col.append(response_col)
    # filter related columns only
    df_filter = df.filter(items=col)
    resp_count = df[response_col].count()
    resp_mean = df[response_col].mean()

    for index in range(len(pred_cont_col)):
        bin_num = 5
        c1 = pred_cont_col[index]

        index2 = index + 1
        if index2 < len(pred_cont_col):
            for index2 in range(index2, len(pred_cont_col)):
                cont_wmr_dic = {}
                c2 = pred_cont_col[index2]

                pair_name = c1 + " and " + c2
                df_mean = (
                    df_filter.groupby(
                        [pd.cut(df_filter[c1], bin_num), pd.cut(df_filter[c2], bin_num)]
                    )
                    .mean()
                    .unstack()
                )
                df_mean_weight = df_mean.copy()

                max = np.max(df_filter[c1])
                min = np.min(df_filter[c1])
                y = np.linspace(min, max, 6)

                max = np.max(df_filter[c2])
                min = np.min(df_filter[c2])
                x = np.linspace(min, max, 6)

                # bin count
                df__bin_count = (
                    df_filter.groupby(
                        [pd.cut(df_filter[c1], bin_num), pd.cut(df_filter[c2], bin_num)]
                    )
                    .count()
                    .unstack()
                )
                df_bin_wt = df__bin_count.copy()
                # population proportion
                df__bin_count[response_col] = df__bin_count[response_col] / resp_count

                cont_wmr_dic["Predictor 1"] = c1
                cont_wmr_dic["Predictor 2"] = c2

                df_mean_new = df_mean[[response_col]].copy()
                # calculate mwr
                df_mean_new[response_col] = (df_mean[response_col] - resp_mean) ** 2
                cont_wmr_dic["Difference of Mean Response"] = df_mean_new[
                    response_col
                ].sum().sum() / (len(x) * len(y))

                df_mean_mwr_w = df_mean.copy()
                # weight
                df_bin_wt[response_col] = df_bin_wt[response_col] / resp_count
                df_mean_mwr_w[response_col] = (
                    (df_mean_mwr_w[response_col] - resp_mean) ** 2
                ) * df_bin_wt[response_col]
                cont_wmr_dic["Weighted Difference of Mean Response"] = (
                    df_mean_mwr_w[response_col].sum().sum()
                )

                # unweighted plot
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=df_mean[response_col],
                        text=df_mean[response_col],
                        texttemplate="%{text:.3f}",
                        colorscale="rdylgn",
                        customdata=df__bin_count[response_col],
                    )
                )
                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "X: %{x}",
                            "Y: %{y}",
                            "Z: %{z}",
                            "Population Proportion: %{customdata:.3f}",
                        ]
                    )
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c2, c1),
                    xaxis_title=c2,
                    yaxis_title=c1,
                )
                # fig.show()
                fig.write_html(
                    file="%s/midtermplots/%s_bin_uw_plot.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cont_wmr_dic["Bin Plot"] = pair_name.replace(" ", "_") + "_bin_uw_plot"

                # weighted plot
                df_mean_weight[response_col] = df_mean_weight[response_col] - resp_mean
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=df_mean_weight[response_col],
                        text=df_mean_weight[response_col],
                        texttemplate="%{text:.3f}",
                        colorscale="blugrn",
                        customdata=df__bin_count[response_col],
                    )
                )
                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "X: %{x}",
                            "Y: %{y}",
                            "Z: %{z}",
                            "Population Proportion: %{customdata:.3f}",
                        ]
                    )
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c2, c1),
                    xaxis_title=c2,
                    yaxis_title=c1,
                )
                # fig.show()
                fig.write_html(
                    file="%s/midtermplots/%s_dwm_of_resp_residual.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cont_wmr_dic["Residual Plot"] = (
                    pair_name.replace(" ", "_") + "_dwm_of_resp_residual"
                )
                cont_cont_mwr_list.append(cont_wmr_dic)

    cont_cont_mwr_list = sorted(
        cont_cont_mwr_list,
        key=lambda x: x["Weighted Difference of Mean Response"],
        reverse=True,
    )
    cont_cont_mwru_df = pd.DataFrame.from_dict(cont_cont_mwr_list)
    if "Bin Plot" in cont_cont_mwru_df.columns:
        cont_cont_mwru_df["Bin Plot"] = cont_cont_mwru_df["Bin Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )
    if "Residual Plot" in cont_cont_mwru_df.columns:
        cont_cont_mwru_df["Residual Plot"] = cont_cont_mwru_df["Residual Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return cont_cont_mwru_df


def cal_cont_cat_mwr(df, pred_cont_col, pred_cat_col, response_col):
    resp_count = df[response_col].count()
    resp_mean = df[response_col].mean()
    cont_cat_mwr_list = []
    bin_num = 9
    for c1 in pred_cont_col:
        for c2 in pred_cat_col:
            cont_cat_mwr_dic = {}
            pair_name = c1 + " and " + c2

            col = []
            col.append(c1)
            col.append(c2)
            col.append(response_col)
            df_filter = df.filter(items=col)

            df_mean = (
                df_filter.groupby([df_filter[c2], pd.cut(df_filter[c1], bin_num)])
                .mean()
                .unstack()
            )
            df_mean_weight = df_mean.copy()

            max = np.max(df_filter[c1])
            min = np.min(df_filter[c1])
            x = np.linspace(min, max, 6)
            y = list(set(df_filter[c2].tolist()))

            cont_cat_mwr_dic["Predictor 1"] = c1
            cont_cat_mwr_dic["Predictor 2"] = c2

            df_mean_new = df_mean[[response_col]].copy()
            df_mean_new[response_col] = (df_mean_new[response_col] - resp_mean) ** 2
            cont_cat_mwr_dic["Difference of Mean Response"] = df_mean_new[
                response_col
            ].sum().sum() / (len(x) * len(list(set(df[c2].tolist()))))

            df_count = (
                df_filter.groupby([df_filter[c2], pd.cut(df_filter[c1], bin_num)])
                .count()
                .unstack()
            )
            df_bin_wt = df_count.copy()
            # calculate population proportion
            df_count[response_col] = df_count[response_col] / resp_count

            df_mean_mwr_w = df_mean.copy()
            # calculate weighted
            df_bin_wt[response_col] = df_bin_wt[response_col] / resp_count
            df_mean_mwr_w[response_col] = (
                (df_mean_mwr_w[response_col] - resp_mean) ** 2
            ) * df_bin_wt[response_col]
            cont_cat_mwr_dic["Weighted Difference of Mean Response"] = (
                df_mean_mwr_w[response_col].sum().sum()
            )

            # unweighted plot
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=df_mean[response_col],
                    text=df_mean[response_col],
                    texttemplate="%{text:.3f}",
                    colorscale="rdylgn",
                    customdata=df_count[response_col],
                )
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "X: %{x}",
                        "Y: %{y}",
                        "Z: %{z}",
                        "Population Proportion: %{customdata:.3f}",
                    ]
                )
            )
            fig.update_layout(
                title="%s and %s Bin Average Response" % (c1, c2),
                xaxis_title=c1,
                yaxis_title=c2,
            )
            # fig.show()
            fig.write_html(
                file="%s/midtermplots/%s_diff_of_mean_resp_bin.html"
                % (rootpath, pair_name.replace(" ", "_")),
                include_plotlyjs="cdn",
            )

            cont_cat_mwr_dic["Bin Plot"] = (
                pair_name.replace(" ", "_") + "_diff_of_mean_resp_bin"
            )

            # weighted plot
            df_mean_weight[response_col] = df_mean_weight[response_col] - resp_mean
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=x,
                    y=y,
                    z=df_mean_weight[response_col],
                    text=df_mean_weight[response_col],
                    texttemplate="%{text:.3f}",
                    colorscale="blugrn",
                    customdata=df_count[response_col],
                )
            )
            fig.update_traces(
                hovertemplate="<br>".join(
                    [
                        "X: %{x}",
                        "Y: %{y}",
                        "Z: %{z}",
                        "Population Proportion: %{customdata:.3f}",
                    ]
                )
            )
            fig.update_layout(
                title="%s and %s Bin Average Response" % (c2, c1),
                xaxis_title=c1,
                yaxis_title=c2,
            )
            # fig.show()
            fig.write_html(
                file="%s/midtermplots/%s_dwm_of_resp_residual.html"
                % (rootpath, pair_name.replace(" ", "_")),
                include_plotlyjs="cdn",
            )

            cont_cat_mwr_dic["Residual Plot"] = (
                pair_name.replace(" ", "_") + "_dwm_of_resp_residual"
            )
            cont_cat_mwr_list.append(cont_cat_mwr_dic)

    cont_cat_mwr_list = sorted(
        cont_cat_mwr_list,
        key=lambda x: x["Weighted Difference of Mean Response"],
        reverse=True,
    )
    con_cat_mwr_df = pd.DataFrame.from_dict(cont_cat_mwr_list)

    if "Bin Plot" in con_cat_mwr_df.columns:
        con_cat_mwr_df["Bin Plot"] = con_cat_mwr_df["Bin Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )
    if "Residual Plot" in con_cat_mwr_df.columns:
        con_cat_mwr_df["Residual Plot"] = con_cat_mwr_df["Residual Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return con_cat_mwr_df


def cal_cat_cat_mwr(df, pred_cat_col, response_col):
    resp_count = df[response_col].count()
    resp_mean = df[response_col].mean()
    cat_cat_mwr_list = []

    for index in range(0, len(pred_cat_col)):
        c1 = pred_cat_col[index]
        index2 = index + 1
        if index2 < len(pred_cat_col):
            for index2 in range(index2, len(pred_cat_col)):
                c2 = pred_cat_col[index2]
                cat_cat_mwr_dic = {}
                pair_name = c1 + " and " + c2

                col = []
                col.append(c1)
                col.append(c2)
                col.append(response_col)
                df_filter = df.filter(items=col)

                df_mean = (
                    df_filter.groupby([df_filter[c1], df_filter[c2]]).mean().unstack()
                )
                df_mean_weight = df_mean.copy()

                cat_cat_mwr_dic["Predictor 1"] = c1
                cat_cat_mwr_dic["Predictor 2"] = c2

                df_mean_new = df_mean[[response_col]].copy()
                df_mean_new[response_col] = (df_mean[response_col] - resp_mean) ** 2
                cat_cat_mwr_dic["Difference of Mean Response"] = df_mean_new[
                    response_col
                ].sum().sum() / (
                    len(list(set(df_filter[c1].tolist())))
                    * len(list(set(df_filter[c2].tolist())))
                )

                # calculate population proportion
                df_count = (
                    df_filter.groupby([df_filter[c1], df_filter[c2]]).count().unstack()
                )
                df_bin_wt = df_count.copy()
                df_count[response_col] = df_count[response_col] / resp_count

                # calculate weighted
                df_mean_mwr_w = df_mean.copy()
                df_bin_wt[response_col] = df_bin_wt[response_col] / resp_count
                df_mean_mwr_w[response_col] = (
                    (df_mean_mwr_w[response_col] - resp_mean) ** 2
                ) * df_bin_wt[response_col]
                cat_cat_mwr_dic["Weighted Difference of Mean Response"] = (
                    df_mean_mwr_w[response_col].sum().sum()
                )

                # plot unweighted
                y = list(set(df[c1].tolist()))
                x = list(set(df_filter[c2].tolist()))
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=df_mean[response_col],
                        text=df_mean[response_col],
                        texttemplate="%{text:.3f}",
                        colorscale="rdylgn",
                        customdata=df_count[response_col],
                    )
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c1, c2),
                    xaxis_title=c1,
                    yaxis_title=c2,
                )
                # fig.show()
                fig.write_html(
                    file="%s/midtermplots/%s_diff_of_mean_resp_bin.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cat_cat_mwr_dic["Bin Plot"] = (
                    pair_name.replace(" ", "_") + "_diff_of_mean_resp_bin"
                )

                # weighted plot
                df_mean_weight[response_col] = df_mean_weight[response_col] - resp_mean
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x=x,
                        y=y,
                        z=df_mean_weight[response_col],
                        text=df_mean_weight[response_col],
                        texttemplate="%{text:.3f}",
                        colorscale="blugrn",
                        customdata=df_count[response_col],
                    )
                )
                fig.update_traces(
                    hovertemplate="<br>".join(
                        [
                            "X: %{x}",
                            "Y: %{y}",
                            "Z: %{z}",
                            "Population Proportion: %{customdata:.3f}",
                        ]
                    )
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c2, c1),
                    xaxis_title=c1,
                    yaxis_title=c2,
                )
                # fig.show()
                fig.write_html(
                    file="%s/midtermplots/%s_dwm_of_resp_residual.html"
                    % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cat_cat_mwr_dic["Residual Plot"] = (
                    pair_name.replace(" ", "_") + "_dwm_of_resp_residual"
                )

                cat_cat_mwr_list.append(cat_cat_mwr_dic)

    cat_cat_mwr_list = sorted(
        cat_cat_mwr_list,
        key=lambda x: x["Weighted Difference of Mean Response"],
        reverse=True,
    )
    cat_cat_mwr_df = pd.DataFrame.from_dict(cat_cat_mwr_list)

    if "Bin Plot" in cat_cat_mwr_df.columns:
        cat_cat_mwr_df["Bin Plot"] = cat_cat_mwr_df["Bin Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )
    if "Residual Plot" in cat_cat_mwr_df.columns:
        cat_cat_mwr_df["Residual Plot"] = cat_cat_mwr_df["Residual Plot"].apply(
            lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(
                urlpath, x, x
            )
        )

    return cat_cat_mwr_df


def create_final_report(
    df1=None,
    cm1=None,
    mwrdf1=None,
    df2=None,
    cm2=None,
    mwrdf2=None,
    df3=None,
    cm3=None,
    mwrdf3=None,
):
    # to open/create a new html file in the write mode
    f = open(rootpath + "/midterm_report.html", "w")

    html_template = """
    """
    if df1 is not None and df2 is not None and df3 is not None:
        html_template = """<html>
        <head>
        <title>report</title>
        <style>
        table {
            margin-left: auto;
            margin-right: auto
        }
        table, th, td {
            border-collapse: collapse;
            padding: 12px;
        }
        table, th {background-color: #dcf2f9; text-align: center}
        table, td {text-align: left}
        table tr:nth-child(even) {
            background-color: #f2f2f2;;
        }
        table tr:nth-child(odd) {
                background-color: white;
            }
        </style>
        </head>
        <body style='max-width: max-content;margin: auto'>
        <h1 style='text-align: center;margin-top:35px'>Midterm</h2>
        <h2 style='text-align: center;'>Continous/Continous Predictor Pairs</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div></div>
        <h3 style='text-align: center;'>Correlation Matrix</h3>
        <div style='max-width: max-content;margin: auto'>
        <iframe marginwidth="35" marginheight="0" scrolling="no"
            src=%s
            width="1400" height="400" frameborder="0" align="middle">
        </iframe>
        </div>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>

        <h2 style='text-align: center;'>Continous/Categorical Predictor Pairs</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div></div>
        <h3 style='text-align: center;'>Correlation Matrix</h3>
        <iframe marginwidth="35" marginheight="15" scrolling="no"
            src=%s
            width="1400" height="500" frameborder="0" align="middle">
        </iframe>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>

        <h2 style='text-align: center;'>Categorical/Categorical Predictor Pairs</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div></div>
        <h3 style='text-align: center;'>Correlation Matrix</h3>
        <iframe marginwidth="35" marginheight="0" scrolling="no"
            src=%s
            width="1400" height="700" frameborder="0" align="middle">
        </iframe>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>
        <br/><br/>
        </body>
        </html>
        """ % (
            df1.to_html(escape=False, index=False, classes="table table-stripped")
            if df1 is not None and not df1.empty
            else "",
            cm1,
            mwrdf1.to_html(escape=False, index=False, classes="table table-stripped")
            if mwrdf1 is not None and not mwrdf1.empty
            else "",
            df2.to_html(escape=False, index=False, classes="table table-stripped")
            if df2 is not None and not df2.empty
            else "",
            cm2,
            mwrdf2.to_html(escape=False, index=False, classes="table table-stripped")
            if mwrdf2 is not None and not mwrdf2.empty
            else "",
            df3.to_html(escape=False, index=False, classes="table table-stripped")
            if df3 is not None and not df3.empty
            else "",
            cm3,
            mwrdf3.to_html(escape=False, index=False, classes="table table-stripped")
            if mwrdf3 is not None and not mwrdf3.empty
            else "",
        )
    elif df1 is not None:
        html_template = """<html>
        <head>
        <title>report</title>
        <style>
        table {
            margin-left: auto;
            margin-right: auto
        }
        table, th, td {
            border-collapse: collapse;
            padding: 12px;
        }
        table, th {background-color: #dcf2f9; text-align: center}
        table, td {text-align: left}
        table tr:nth-child(even) {
            background-color: #f2f2f2;;
        }
        table tr:nth-child(odd) {
                background-color: white;
            }
        </style>
        </head>
        <body style='max-width: max-content;margin: auto'>
        <h1 style='text-align: center;margin-top:35px'>Midterm</h2>
        <h2 style='text-align: center;'>Continous/Continous Predictor Pairs</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div></div>
        <h3 style='text-align: center;'>Correlation Matrix</h3>
        <div style='max-width: max-content;margin: auto'>
        <iframe marginwidth="35" marginheight="0" scrolling="no"
            src=%s
            width="1400" height="400" frameborder="0" align="middle">
        </iframe>
        </div>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>
        <br/><br/>
        </body>
        </html>
        """ % (
            df1.to_html(escape=False, index=False, classes="table table-stripped")
            if df1 is not None and not df1.empty
            else "",
            cm1,
            mwrdf1.to_html(escape=False, index=False, classes="table table-stripped")
            if mwrdf1 is not None and not mwrdf1.empty
            else "",
        )
    elif df3 is not None:
        html_template = """<html>
        <head>
        <title>report</title>
        <style>
        table {
            margin-left: auto;
            margin-right: auto
        }
        table, th, td {
            border-collapse: collapse;
            padding: 12px;
        }
        table, th {background-color: #dcf2f9; text-align: center}
        table, td {text-align: left}
        table tr:nth-child(even) {
            background-color: #f2f2f2;;
        }
        table tr:nth-child(odd) {
                background-color: white;
            }
        </style>
        </head>
        <body style='max-width: max-content;margin: auto'>
        <h1 style='text-align: center;margin-top:35px'>Midterm</h2>
        <h2 style='text-align: center;'>Categorical/Categorical Predictor Pairs</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div></div>
        <h3 style='text-align: center;'>Correlation Matrix</h3>
        <div style='max-width: max-content;margin: auto'>
        <iframe marginwidth="35" marginheight="0" scrolling="no"
            src=%s
            width="1400" height="400" frameborder="0" align="middle">
        </iframe>
        </div>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>
        <br/><br/>
        </body>
        </html>
        """ % (
            df3.to_html(escape=False, index=False, classes="table table-stripped")
            if df3 is not None and not df3.empty
            else "",
            cm3,
            mwrdf3.to_html(escape=False, index=False, classes="table table-stripped")
            if mwrdf3 is not None and not mwrdf3.empty
            else "",
        )

    # write code into the file
    f.write(html_template)
    f.close()

    new = 2  # open in a new tab, if possible
    url = "file://%s/midterm_report.html" % rootpath
    webbrowser.open(url, new=new)


def main():
    # make a midtermplots folder if it does not exist
    if not os.path.exists(rootpath + "/midtermplots"):
        os.makedirs(rootpath + "/midtermplots")

    df, predictor_col, response_col = get_test_data_set("titanic")
    # df, predictor_col, response_col = get_test_data_set("tips")
    # df, predictor_col, response_col = get_test_data_set("diabetes")
    df = df.dropna().reset_index(drop=True)

    # split predictors column
    pred_cat_col = []
    pred_cont_col = []
    for column in predictor_col:
        pred_type = determin_type(df, column)
        if pred_type == "cat":
            pred_cat_col.append(column)
        else:
            pred_cont_col.append(column)

    pred_cat_df = df.filter(items=pred_cat_col)
    pred_cont_df = df.filter(items=pred_cont_col)

    # 1. Continuous / Continuous pairs
    if len(pred_cont_col) > 0:
        cont_cont_corr_df = cal_cont_cont_corre(pred_cont_df)

        # correlation matrix
        cont_cont_corr_matrix = pred_cont_df.corr()
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cont_cont_corr_matrix.columns,
                y=cont_cont_corr_matrix.index,
                z=np.array(cont_cont_corr_matrix),
                colorscale="blugrn",
            )
        )
        # fig.show()
        fig.write_html(
            file="%s/midtermplots/corr_matrix_cont_cont.html" % (rootpath),
            include_plotlyjs="cdn",
        )

        cont_cont_cm_plot = "%s/midtermplots/corr_matrix_cont_cont.html" % (rootpath)

        # brute force table
        cont_cont_mwr_df = cal_cont_cont_mwr(df, pred_cont_col, response_col)

    # 2. Continuous / Categorical pairs
    if len(pred_cont_col) > 0 and len(pred_cat_col) > 0:
        cont_cat_corr_df = cal_cont_cat_corre(pred_cont_df, pred_cat_df, df)

        # correlation matrix
        rows = []
        for v1 in pred_cont_df.columns:
            col = []
            for v2 in pred_cat_df.columns:
                corr = cat_cont_correlation_ratio(
                    pred_cat_df[v2].values, pred_cont_df[v1].values
                )
                col.append(corr)
            rows.append(col)

        results = np.array(rows)
        cont_cat_corr_matrix = pd.DataFrame(
            results, columns=pred_cat_df.columns, index=pred_cont_df.columns
        )

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cont_cat_corr_matrix.columns,
                y=cont_cat_corr_matrix.index,
                z=np.array(cont_cat_corr_matrix),
                colorscale="blugrn",
            )
        )
        # fig.show()
        fig.write_html(
            file="%s/midtermplots/corr_matrix_cont_cat.html" % (rootpath),
            include_plotlyjs="cdn",
        )

        cont_cat_cm_plot = "%s/midtermplots/corr_matrix_cont_cat.html" % (rootpath)

        # brute force table
        cont_cat_mwr_df = cal_cont_cat_mwr(
            df, pred_cont_col, pred_cat_col, response_col
        )

    # 3. Categorical / Categorical pairs
    if len(pred_cat_col) > 0:
        cat_cat_corr_df = cal_cat_cat_corre(pred_cat_df)

        # correlation matrix
        rows = []
        for v1 in pred_cat_df.columns:
            col = []
            for v2 in pred_cat_df.columns:
                corr = cat_correlation(pred_cat_df[v1].values, pred_cat_df[v2].values)
                col.append(corr)
            rows.append(col)

        results = np.array(rows)
        cat_cat_corr_matrix = pd.DataFrame(
            results, columns=pred_cat_df.columns, index=pred_cat_df.columns
        )
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x=cat_cat_corr_matrix.columns,
                y=cat_cat_corr_matrix.index,
                z=np.array(cat_cat_corr_matrix),
                colorscale="blugrn",
            )
        )
        # fig.show()
        fig.write_html(
            file="%s/midtermplots/corr_matrix_cat_cat.html" % (rootpath),
            include_plotlyjs="cdn",
        )
        cat_cat_cm_plot = "%s/midtermplots/corr_matrix_cat_cat.html" % (rootpath)

        # brute force table
        cat_cat_mwr_df = cal_cat_cat_mwr(df, pred_cat_col, response_col)

    # create report
    if len(pred_cont_col) > 0 and len(pred_cat_col) > 0:
        create_final_report(
            cont_cont_corr_df,
            cont_cont_cm_plot,
            cont_cont_mwr_df,
            cont_cat_corr_df,
            cont_cat_cm_plot,
            cont_cat_mwr_df,
            cat_cat_corr_df,
            cat_cat_cm_plot,
            cat_cat_mwr_df,
        )
    elif len(pred_cont_col) > 0:
        create_final_report(cont_cont_corr_df, cont_cont_cm_plot, cont_cont_mwr_df)
    elif len(pred_cat_col) > 0:
        create_final_report(
            None,
            None,
            None,
            None,
            None,
            None,
            cat_cat_corr_df,
            cat_cat_cm_plot,
            cat_cat_mwr_df,
        )
    return


if __name__ == "__main__":
    sys.exit(main())

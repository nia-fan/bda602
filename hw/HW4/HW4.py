import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf
from IPython.display import HTML
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

rootpath = os.path.dirname(sys.argv[0])
urlpath = "https://nia-fan.github.io/plots"


def check_distinct(df, col):
    unique = list(set(df[col].tolist()))
    return len(unique)


def determine_type(df, col):
    col_type = df[col].dtype
    if col_type == "object" or col_type == "bool":
        # Convert Categorical Data Columns to Numerical
        le = LabelEncoder()
        label_encd = le.fit_transform(df[col])
        df[col] = label_encd
        return "cat"
    else:
        # check distinct values
        len_dist = check_distinct(df, col)
        if len_dist / len(df) < 0.05:
            return "cat"
        else:
            return "cont"


def plot(resp_type, predi_type, res_value, pred_value, feature_name, data_dic):
    if resp_type == "cat":
        x = res_value
        y = pred_value

        # Categorical Response vs Categorical Predictor
        if predi_type == "cat":
            conf_matrix = confusion_matrix(x, y)
            fig_cat = go.Figure(
                data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
            )
            fig_cat.update_layout(
                title="Categorical Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor %s" % feature_name,
            )
            # fig_cat.show()

            fig_cat.write_html(
                file="%s/plots/cat_resp_cat_pred_%s_heatmap_plot.html"
                % (rootpath, feature_name),
                include_plotlyjs="cdn",
            )

            data_dic["plot"] = "%s/cat_resp_cat_pred_%s_heatmap_plot.html" % (
                urlpath,
                feature_name,
            )

        # Categorical Response vs Continuous Predictor
        elif predi_type == "cont":
            group_labels = list(set(x))

            # violin plot
            fig_violin = go.Figure()
            for curr_group in group_labels:
                fig_violin.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(y)),
                        y=y,
                        name=str(curr_group),
                        box_visible=True,
                        meanline_visible=True,
                    )
                )

            fig_violin.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Response",
                yaxis_title="Predictor %s" % feature_name,
            )
            # fig_violin.show()

            fig_violin.write_html(
                file="%s/plots/cat_resp_cont_pre_%s_violin_plot.html"
                % (rootpath, feature_name),
                include_plotlyjs="cdn",
            )

            # this case has two plots
            plot_url = "%s/cat_resp_cont_pre_%s_violin_plot.html" % (
                urlpath,
                feature_name,
            )

            # distribution plot
            hist_data = [y]
            fig_dist = ff.create_distplot(hist_data, [feature_name], bin_size=0.2)
            fig_dist.update_layout(
                title="Continuous Predictor by Categorical Response",
                xaxis_title="Predictor %s" % feature_name,
                yaxis_title="Distribution",
            )
            # fig_dist.show()
            fig_dist.write_html(
                file="%s/plots/cat_resp_cont_pre_%s_dist_plot.html"
                % (rootpath, feature_name),
                include_plotlyjs="cdn",
            )

            plot_url = (
                plot_url
                + " , "
                + "%s/cat_resp_cont_pre_%s_dist_plot.html" % (urlpath, feature_name)
            )
            data_dic["plot"] = plot_url
    else:
        # Continuous Response vs Categorical Predictor
        if predi_type == "cat":
            # violin plot
            group_labels = list(set(res_value))
            fig_violin_2 = go.Figure()
            for curr_group in group_labels:
                fig_violin_2.add_trace(
                    go.Violin(
                        x=np.repeat(curr_group, len(pred_value)),
                        y=pred_value,
                        name=curr_group,
                        box_visible=True,
                        meanline_visible=True,
                    )
                )
                fig_violin_2.update_layout(
                    title="Continuous Response by Categorical Predictor",
                    xaxis_title="Predictor %s" % feature_name,
                    yaxis_title="Response",
                )
                # fig_violin_2.show()

                fig_violin_2.write_html(
                    file="%s/plots/cont_resp_cat_pre_%s_violin_plot.html"
                    % (rootpath, feature_name),
                    include_plotlyjs="cdn",
                )
                plot_url2 = "%s/cont_resp_cat_pre_%s_violin_plot.html" % (
                    urlpath,
                    feature_name,
                )

                # Create distribution plot
                fig_dis_2 = ff.create_distplot(
                    [pred_value], [feature_name], bin_size=0.2
                )
                fig_dis_2.update_layout(
                    title="Continuous Response by Categorical Predictor",
                    xaxis_title="Response",
                    yaxis_title="Distribution",
                )
                # fig_dis_2.show()

                fig_dis_2.write_html(
                    file="%s/plots/cont_resp_cat_pre_%s_dist_plot.html"
                    % (rootpath, feature_name),
                    include_plotlyjs="cdn",
                )

                plot_url2 = (
                    plot_url2
                    + " , "
                    + "%s/cat_resp_cont_pre_%s_dist_plot.html" % (urlpath, feature_name)
                )

        # Continuous Response vs Continuous Predictor
        elif predi_type == "cont":
            x = pred_value
            y = res_value
            # Continuous Response by Continuous Predictor
            fig_scatt = px.scatter(x=x, y=y, trendline="ols")
            fig_scatt.update_layout(
                title="Continuous Response by Continuous Predictor",
                xaxis_title="Predictor %s" % feature_name,
                yaxis_title="Response",
            )
            # fig_scatt.show()

            fig_scatt.write_html(
                file="%s/plots/cont_resp_cont_pre_%s_scatter_plot.html"
                % (rootpath, feature_name),
                include_plotlyjs="cdn",
            )

            data_dic["plot"] = "%s/cont_resp_cont_pre_%s_scatter_plot.html" % (
                urlpath,
                feature_name,
            )


def calculate_stats(
    resp_type, res_value, pred_value, feature_name, response_name, df, data_dic
):
    # Regression: Continuous response
    if resp_type == "cont":
        x = pred_value
        # add constant to predictor variables
        x = statsmodels.api.add_constant(x)

        # fit linear regression model
        linear_regres_modal = statsmodels.api.OLS(res_value, x)
        linear_regres_modal_fit = linear_regres_modal.fit()
        # print(linear_regres_modal_fit.summary())

        # Get the stats
        t_value = round(linear_regres_modal_fit.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regres_modal_fit.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=pred_value, y=res_value, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="response",
        )
        # fig.show()

        fig.write_html(
            file="%s/plots/linear_regres_%s_plot.html" % (rootpath, feature_name),
            include_plotlyjs="cdn",
        )

        data_dic["t-score"] = "%s/linear_regres_%s_plot.html" % (urlpath, feature_name)
        data_dic["p-value"] = "%s/linear_regres_%s_plot.html" % (urlpath, feature_name)

    # Logistic Regression: Boolean response
    elif resp_type == "cat":
        log_reg = smf.logit("%s ~ %s" % (response_name, feature_name), data=df).fit()

        # print(log_reg.summary())
        t_value2 = log_reg.tvalues[1]
        p_value2 = log_reg.pvalues[1]

        # Plot the figure
        fig2 = px.scatter(x=pred_value, y=df[response_name].values, trendline="ols")
        fig2.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value2}) (p-value={p_value2})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="response",
        )
        # fig2.show()

        fig2.write_html(
            file="%s/plots/scatter_%s_plot.html" % (rootpath, feature_name),
            include_plotlyjs="cdn",
        )

        data_dic["t-score"] = "%s/scatter_%s_plot.html" % (urlpath, feature_name)
        data_dic["p-value"] = "%s/scatter_%s_plot.html" % (urlpath, feature_name)


def plot_MWR_unweighted(res_value, pred_value, feature_name, data_dic, df):
    diff = [(pred_value[i] - res_value[i]) for i in range(len(pred_value))]
    diff_list = np.array(diff)

    max = np.max(pred_value)
    min = np.min(pred_value)
    bins = np.linspace(min, max, 10)

    df["bin"] = pd.cut(diff_list, bins)
    agg_df = df.groupby(by="bin").mean()
    mids = pd.IntervalIndex(agg_df.index.get_level_values("bin")).mid

    fig_mwr = go.Figure()
    fig_mwr = make_subplots(specs=[[{"secondary_y": True}]])

    fig_mwr.add_trace(go.Histogram(x=pred_value, name="Population"))
    fig_mwr.add_trace(
        go.Scatter(x=bins, y=res_value, mode="lines", line_color="green", name="μpop"),
        secondary_y=False,
    )
    fig_mwr.add_trace(
        go.Scatter(x=bins, y=mids, mode="lines", line_color="red", name="μi−μpop"),
        secondary_y=True,
    )

    # Set titles
    fig_mwr.update_yaxes(title_text="Response", secondary_y=False)
    fig_mwr.update_yaxes(title_text="Population", secondary_y=True)

    fig_mwr.update_layout(
        title="Binned Difference with Mean of Response vs Bin Unweighted (%s)"
        % feature_name,
        barmode="group",
        bargap=0.30,
        bargroupgap=0.0,
    )

    fig_mwr.add_annotation(
        dict(
            x=0.5,
            y=-0.06,
            showarrow=False,
            text="Predictor Bin",
            xref="paper",
            yref="paper",
        )
    )

    # fig_mwr.show()
    fig_mwr.write_html(
        file="%s/plots/MWR_unw_%s_plot.html" % (rootpath, feature_name),
        include_plotlyjs="cdn",
    )

    data_dic["MWR Unweighted"] = "%s/MWR_unw_%s_plot.html" % (urlpath, feature_name)


def plot_MWR_weighted(res_value, pred_value, feature_name, data_dic, df):
    diff = [(pred_value[i] - res_value[i]) for i in range(len(pred_value))]
    diff_list = np.array(diff)

    steps = np.linspace(np.percentile(pred_value, 5), np.percentile(pred_value, 96), 8)
    np.insert(steps, 0, np.percentile(pred_value, 6))
    np.append(steps, np.percentile(pred_value, 95))
    bins = steps

    df["bin"] = pd.cut(diff_list, bins)
    agg_df = df.groupby(by="bin").mean()
    mids = pd.IntervalIndex(agg_df.index.get_level_values("bin")).mid

    fig_mwr_w = go.Figure()
    fig_mwr_w = make_subplots(specs=[[{"secondary_y": True}]])

    fig_mwr_w.add_trace(go.Histogram(x=pred_value, name="Population"))
    fig_mwr_w.add_trace(
        go.Scatter(x=bins, y=res_value, mode="lines", line_color="green", name="μpop"),
        secondary_y=False,
    )
    fig_mwr_w.add_trace(
        go.Scatter(x=bins, y=mids, mode="lines", line_color="red", name="μi−μpop"),
        secondary_y=True,
    )

    # Set y-axes titles
    fig_mwr_w.update_yaxes(title_text="Response", secondary_y=False)
    fig_mwr_w.update_yaxes(title_text="Population", secondary_y=True)

    fig_mwr_w.update_layout(
        title="Binned Difference with Mean of Response vs Bin Weighted (%s)"
        % feature_name,
        barmode="group",
        bargap=0.30,
        bargroupgap=0.0,
    )

    fig_mwr_w.add_annotation(
        dict(
            x=0.5,
            y=-0.06,
            showarrow=False,
            text="Predictor Bin",
            xref="paper",
            yref="paper",
        )
    )
    # fig_mwr_w.show()

    fig_mwr_w.write_html(
        file="%s/plots/MWR_wt_%s_plot.html" % (rootpath, feature_name),
        include_plotlyjs="cdn",
    )

    data_dic["MWR Weighted"] = "%s/MWR_wt_%s_plot.html" % (urlpath, feature_name)


def random_forest_ranking(df_pred_cont):
    # Create training and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        df_pred_cont.iloc[:, 1:], df_pred_cont.iloc[:, 0], test_size=0.3, random_state=1
    )
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)

    rfc = RandomForestClassifier(n_estimators=500, random_state=1)
    rfc.fit(X_train_std, y_train.values.ravel())

    importances = rfc.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    feat_labels = df_pred_cont.columns[:-1]
    ranking_df = pd.DataFrame(columns=["rank", "feature_name", "score"])
    for f in range(X_train.shape[1]):
        ranking_df.loc[len(ranking_df.index)] = [
            f + 1,
            feat_labels[sorted_indices[f]],
            importances[sorted_indices[f]],
        ]

    x = list(df_pred_cont.columns.values)
    fig_bar = go.Figure(data=go.Bar(x=x, y=importances))
    # fig.show()
    fig_bar.write_html(
        file="%s/plots/rf_feature_importance.html" % (rootpath),
        include_plotlyjs="cdn",
    )


def main():
    # make a plots folder if it does not exist
    if not os.path.exists(rootpath + "/plots"):
        os.makedirs(rootpath + "/plots")

    # 1. get data from sklearn dtasets
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    )
    df = df.dropna().reset_index(drop=True)

    response_col = "survived"
    Predicator_col = [x for x in df.columns if x != "survived"]

    # 2. check response type
    resp_type = determine_type(df, response_col)
    response_name = ""
    if resp_type == "cat":
        response_name = "Response (boolean)"
    else:
        response_name = "Response (continuous)"

    # define result data list
    results_data = []

    # 3. Loop through each predictor column
    predic_type_cont = ["survived"]
    for column in Predicator_col:
        data_dic = {}
        data_dic[response_name] = response_col

        # determine predictor column type
        predi_type = determine_type(df, column)
        pre_name = ""
        if predi_type == "cat":
            pre_name = "%s (cat)" % column
        else:
            pre_name = "%s (cont)" % column
            predic_type_cont.append(column)

        data_dic["Predictor"] = pre_name

        # generate plot
        plot(
            resp_type,
            predi_type,
            df[response_col].values,
            df[column].values,
            column,
            data_dic,
        )

        # calculate p-values and t-scores (continuous predictors only)
        if predi_type == "cont":
            calculate_stats(
                resp_type,
                df[response_col].values,
                df[column].values,
                column,
                response_col,
                df,
                data_dic,
            )
        else:
            data_dic["t-score"] = "N/A"
            data_dic["p-value"] = "N/A"

        # Difference with mean of response along with its plot (weighted and unweighted)
        plot_MWR_unweighted(
            df[response_col].values, df[column].values, column, data_dic, df
        )
        plot_MWR_weighted(
            df[response_col].values, df[column].values, column, data_dic, df
        )

        results_data.append(data_dic)

    # for some reason, sklearn RandomForestClassifier did not work for individual columns, so run whole df
    df_pred_cont = df.drop(columns=df.columns.difference(predic_type_cont))
    random_forest_ranking(df_pred_cont)

    # set RF VarImp in results_data
    for item in results_data:
        item["RF VarImp"] = "%s/plots/rf_feature_importance.html" % (urlpath)

    result_df = pd.DataFrame.from_dict(results_data)

    # add response and predictor info
    index = len(result_df.index)
    result_df.loc[index + 1] = ["", "", "df", "", "", "", "", ""]
    result_df.loc[index + 2] = ["", "", "response=survived", "", "", "", "", ""]
    pred_str = ",".join(str(x) for x in Predicator_col)
    result_df.loc[index + 3] = [
        "",
        "",
        "predictors=[%s]" % (pred_str),
        "",
        "",
        "",
        "",
        "",
    ]
    HTML(result_df.to_html(escape=False))
    # HTML(result_df.to_html(classes='table table-stripped', escape=False, render_links=True))

    result_df.to_csv("%s/report.csv" % (rootpath), index=None, header=True)

    # write html to file
    result_df.to_html(
        "%s/report.html" % (rootpath),
        classes="table table-striped",
        escape=False,
        render_links=True,
    )
    print(result_df.head())

    return


if __name__ == "__main__":
    sys.exit(main())

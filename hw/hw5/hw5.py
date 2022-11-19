import os
import sys
import webbrowser

import calculate_correlation as calculate_correlation
import findspark
import numpy as np
import pandas as pd
import statsmodels.api
import statsmodels.formula.api as smf
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

findspark.init()
# use below line and change path if path needed for spark
# findspark.init("/Users/nia/Spark/spark-3.3.0-bin-hadoop3", edit_rc=True)
rootpath = os.path.dirname(sys.argv[0])
urlpath = "%s/plots/" % (rootpath)


def determin_type(df, col):
    col_type = df[col].dtype
    if col_type == "object" or col_type == "bool":
        return "cat"
    else:
        return "cont"


def plot(df, pred_col, data_dic, response_col, resp_type):
    # Continuous Response vs Continuous Predictor
    if resp_type == "cont":
        x = df[pred_col]
        y = df[response_col]

        fig_scatt = px.scatter(x=x, y=y, trendline="ols")
        fig_scatt.update_layout(
            title="Continuous Response by Continuous Predictor",
            xaxis_title="Predictor %s" % pred_col,
            yaxis_title="Response",
        )
        # fig_scatt.show()

        fig_scatt.write_html(
            file="%s/plots/cont_resp_cont_pre_%s_scatter_plot.html"
            % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )

        data_dic[
            "plot"
        ] = "<a target='_blank' href='{0}/cont_resp_cont_pre_{1}_scatter_plot.html'>Plot</a>".format(
            urlpath, pred_col
        )
    else:
        x = df[response_col]
        y = df[pred_col]

        group_labels = list(set(x))
        # violin plot
        fig_violin = go.Figure()
        for curr_group in group_labels:
            fig_violin.add_trace(
                go.Violin(
                    x=df[response_col][df[response_col] == curr_group],
                    y=df[pred_col][df[response_col] == curr_group],
                    name=str(curr_group),
                    box_visible=True,
                    meanline_visible=True,
                )
            )
        fig_violin.update_layout(
            title="Continuous Predictor by Categorical Response",
            xaxis_title="Response",
            yaxis_title="Predictor %s" % pred_col,
        )
        # fig_violin.show()
        fig_violin.write_html(
            file="%s/plots/cat_resp_cont_pre_%s_violin_plot.html"
            % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )

        plot_url = "<a target='_blank' href='{0}/cat_resp_cont_pre_{1}_violin_plot.html'>Violin Plot</a>".format(
            urlpath, pred_col
        )

        # distribution plot
        hlist = df[df[response_col] == "H"][pred_col].values
        alist = df[df[response_col] == "A"][pred_col].values
        hist_data = [hlist, alist]
        # print(hist_data)

        fig_dist = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_dist.update_layout(
            title="Continuous Predictor by Categorical Response",
            xaxis_title="Predictor %s" % pred_col,
            yaxis_title="Distribution",
        )
        # fig_dist.show()
        fig_dist.write_html(
            file="%s/plots/cat_resp_cont_pre_%s_dist_plot.html" % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )
        plot_url = (
            plot_url
            + ",&nbsp;&nbsp;<a target='_blank' href='{0}/cat_resp_cont_pre_{1}_dist_plot.html'>Hist Plot</a>".format(
                urlpath, pred_col
            )
        )
        data_dic["plot"] = plot_url


def calculate_stats(pred_col, data_dic, df, res_col, resp_type):
    res_value = df[res_col]
    pred_value = df[pred_col]
    if resp_type == "cont":
        x = pred_value
        # add constant to predictor variables
        x = statsmodels.api.add_constant(x)

        # fit linear regression model
        linear_regres_modal = statsmodels.api.OLS(res_value, x)
        linear_regres_modal_fit = linear_regres_modal.fit()

        # Get the stats
        t_value = round(linear_regres_modal_fit.tvalues[1], 6)
        p_value = "{:.6e}".format(linear_regres_modal_fit.pvalues[1])

        # Plot the figure
        fig = px.scatter(x=pred_value, y=res_value, trendline="ols")
        fig.update_layout(
            title=f"Variable: {res_col}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {res_col}",
            yaxis_title="response",
        )
        # fig.show()
        fig.write_html(
            file="%s/plots/linear_regres_%s_plot.html" % (rootpath, res_col),
            include_plotlyjs="cdn",
        )

        data_dic[
            "t-score"
        ] = "{0}, <a target='_blank' href='{1}/linear_regres_{2}_plot.html'>Plot</a>".format(
            t_value, urlpath, res_col
        )
        data_dic["p-value"] = p_value
    elif resp_type == "cat":
        log_reg = smf.logit("%s ~ %s" % (res_col, pred_col), data=df).fit()
        # print(log_reg.summary())
        t_value2 = log_reg.tvalues[1]
        p_value2 = log_reg.pvalues[1]

        # Plot the figure
        fig2 = px.scatter(x=pred_value, y=df[res_col].values, trendline="ols")
        fig2.update_layout(
            title=f"Variable: {res_col}: (t-value={t_value2}) (p-value={p_value2})",
            xaxis_title=f"Variable: {res_col}",
            yaxis_title="response",
        )
        # fig2.show()
        fig2.write_html(
            file="%s/plots/scatter_%s_plot.html" % (rootpath, res_col),
            include_plotlyjs="cdn",
        )

        data_dic[
            "t-score"
        ] = "{0}, <a target='_blank' href='{1}/scatter_{2}_plot.html'>Plot</a>".format(
            t_value2, urlpath, res_col
        )
        data_dic["p-value"] = p_value2


def cal_MWR(pred_col, res_col, data_dic, df):
    df_filter = df.filter(items=[pred_col, res_col])

    resp_count = df_filter[res_col].count()
    res_value = df_filter[res_col]
    pred_value = df_filter[pred_col]

    max = np.max(pred_value)
    min = np.min(pred_value)
    bin_num = 10
    bins = np.linspace(min, max, bin_num)

    df_mean = df_filter.groupby(pd.cut(df_filter[pred_col], bins)).mean()
    df_mwr = df_mean[[res_col]].copy()
    df_mwr_w = df_mean[[res_col]].copy()
    mwr_uw = ((df_mwr[res_col] - res_value.mean()) ** 2).sum() / len(bins)
    data_dic["MWR Unweighted"] = mwr_uw

    df_bin_wt = df_filter.groupby(pd.cut(df_filter[pred_col], bins)).count()
    df_bin_wt[res_col] = df_bin_wt[res_col] / resp_count
    data_dic["MWR Weighted"] = (
        ((df_mwr_w[res_col] - res_value.mean()) ** 2) * df_bin_wt[res_col]
    ).sum()

    res_mean_lst = []
    for i in range(10):
        res_mean_lst.append(res_value.mean())
    fig_mwr = go.Figure()
    fig_mwr = make_subplots(specs=[[{"secondary_y": True}]])

    fig_mwr.add_trace(go.Histogram(x=pred_value, name="Population"))
    fig_mwr.add_trace(
        go.Scatter(
            x=bins, y=res_mean_lst, mode="lines", line_color="green", name="μpop"
        ),
        secondary_y=False,
    )
    fig_mwr.add_trace(
        go.Scatter(
            x=bins,
            y=(df_mean[pred_col] - res_value.mean()),
            mode="lines",
            line_color="red",
            name="μi−μpop",
        ),
        secondary_y=True,
    )

    # Set titles
    fig_mwr.update_yaxes(title_text="Response", secondary_y=False)
    fig_mwr.update_yaxes(title_text="Population", secondary_y=True)

    fig_mwr.update_layout(
        title="Binned Difference with Mean of Response vs Bin Unweighted (%s)"
        % pred_col,
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
        file="%s/plots/MWR_unw_%s_plot.html" % (rootpath, pred_col),
        include_plotlyjs="cdn",
    )

    data_dic[
        "MWR Unweighted"
    ] = "{0}, <a target='_blank' href='{1}/MWR_unw_{2}_plot.html'>Plot</a>".format(
        mwr_uw, urlpath, pred_col
    )


def random_forest_ranking(df2, pred_cols, response_col):
    resp = df2[response_col]
    pred = df2.filter(items=pred_cols)
    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(
        pred, resp, test_size=0.3, random_state=1
    )
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)

    rfc = RandomForestClassifier(n_estimators=500, random_state=1)
    rfc.fit(X_train_std, y_train)

    importances = rfc.feature_importances_
    return importances


def feature_analy_plot_ranking(df, predictor_cols, response_col, resp_type):
    df2 = df.copy()

    # convert response to numeric
    if resp_type == "cat":
        le = LabelEncoder()
        label_encd = le.fit_transform(df2[response_col])
        df2[response_col] = label_encd

    # define result data list
    results_data = []
    for column in predictor_cols:
        data_dic = {}
        data_dic["Response"] = response_col
        data_dic["Predictor"] = column

        plot(df, column, data_dic, response_col, resp_type)
        calculate_stats(column, data_dic, df2, response_col, resp_type)
        cal_MWR(column, response_col, data_dic, df2)

        results_data.append(data_dic)

    rf_list = random_forest_ranking(df2, predictor_cols, response_col)
    index = 0
    for item in results_data:
        item["RF VarImp"] = rf_list[index]
        index += 1

    results_data = sorted(results_data, key=lambda x: x["RF VarImp"], reverse=True)
    result_df = pd.DataFrame.from_dict(results_data)
    return result_df


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

                r, t_value, p_value = calculate_correlation.cont_cont_correlation(x, y)
                cont_cont_dic["Predictors"] = pair_name
                cont_cont_dic["Pearson's r"] = r
                cont_cont_dic["Absolute Value of Correlation"] = abs(r)

                # Plot the figure
                fig = px.scatter(x=x, y=y, trendline="ols")
                fig.update_layout(
                    title=f"Variable: (t-value={t_value}) (p-value={p_value})",
                    xaxis_title=pred_cont_df.columns[index],
                    yaxis_title=pred_cont_df.columns[index2],
                )
                # fig.show()
                fig.write_html(
                    file="%s/plots/%s_linear_regression.html"
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


def cal_cont_cont_mwr(df, predictor_cols, response_col, resp_type):
    cont_cont_mwr_list = []
    col = predictor_cols[:]
    col.append(response_col)
    # filter only related columns
    df_filter = df.filter(items=col)
    if resp_type == "cat":
        le = LabelEncoder()
        label_encd = le.fit_transform(df_filter[response_col])
        df_filter[response_col] = label_encd

    resp_count = df_filter[response_col].count()
    resp_mean = df_filter[response_col].mean()

    for index in range(len(predictor_cols)):
        bin_num = 5
        c1 = predictor_cols[index]

        index2 = index + 1
        if index2 < len(predictor_cols):
            for index2 in range(index2, len(predictor_cols)):
                cont_wmr_dic = {}
                c2 = predictor_cols[index2]

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
                    file="%s/plots/%s_bin_uw_plot.html"
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
                    file="%s/plots/%s_dwm_of_resp_residual.html"
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


def feature_analy_Correl_BF(df, predictor_cols, response_col, resp_type):
    pred_cont_df = df.filter(items=predictor_cols)
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
        file="%s/plots/corr_matrix_cont_cont.html" % (rootpath),
        include_plotlyjs="cdn",
    )

    cont_cont_cm_plot = "%s/plots/corr_matrix_cont_cont.html" % (rootpath)

    # brute force table
    cont_cont_mwr_df = cal_cont_cont_mwr(df, predictor_cols, response_col, resp_type)

    return (cont_cont_corr_df, cont_cont_cm_plot, cont_cont_mwr_df)


def create_report(pr_df, df1, cm1, mwrdf1, predictor_cols, response_col):
    pred_str = ",".join(str(x) for x in predictor_cols)
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
        <body style='max-width: max-content;margin: 2rem'>
        <h2 style='text-align: center;'>Feature Analyzing Plots and Rankings</h2>
        <h3 style='text-align: center;'>Correlation Table</h3>
        <div>%s</div>
        <div>response=%s</div>
        <div></div>
        <div>predictors=%s</div>
        <br/>
        <h2 style='text-align: center;'>Feature Analyzing  Correlations and Brute-force </h2>
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
        pr_df.to_html(escape=False, index=False, classes="table table-stripped"),
        response_col,
        pred_str,
        df1.to_html(escape=False, index=False, classes="table table-stripped")
        if df1 is not None and not df1.empty
        else "",
        cm1,
        mwrdf1.to_html(escape=False, index=False, classes="table table-stripped")
        if mwrdf1 is not None and not mwrdf1.empty
        else "",
    )

    f = open(rootpath + "/report.html", "w")
    f.write(html_template)
    f.close()

    new = 2  # open in a new tab, if possible
    url = "file://%s/report.html" % rootpath
    webbrowser.open(url, new=new)


def model_compare(df, pred_col, response_col, resp_type):
    df2 = df.copy()
    if resp_type == "cat":
        le = LabelEncoder()
        label_encd = le.fit_transform(df2[response_col])
        df2[response_col] = label_encd

    y = df2[response_col]
    X = df2.filter(items=pred_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Linear Regression
    print("Linear Regression")
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    pred_train_lr = model_lr.predict(X_train)
    print("Training data RMSE", np.sqrt(mean_squared_error(y_train, pred_train_lr)))
    print("Training data R-squared", r2_score(y_train, pred_train_lr))

    pred_test_lr = model_lr.predict(X_test)
    print("Test data RMSE", np.sqrt(mean_squared_error(y_test, pred_test_lr)))
    print("Test data R-squared", r2_score(y_test, pred_test_lr))

    # Random Forest Regression
    rfr = RandomForestRegressor()
    rfr.fit(X_train, y_train)
    pred_train_rfr = rfr.predict(X_train)
    score = rfr.score(X_train, y_train)
    print("Random Forest Regression")
    print("Training data RMSE", np.sqrt(mean_squared_error(y_train, pred_train_rfr)))
    print("Training data R-squared:", score)

    pred_test_rfr = rfr.predict(X_test)
    print("Test data RMSE", np.sqrt(mean_squared_error(y_test, pred_test_rfr)))
    print("Test data R-squared", r2_score(y_test, pred_test_rfr))

    # Elastic Net
    model_enet = ElasticNet()
    model_enet.fit(X_train, y_train)
    pred_train_enet = model_enet.predict(X_train)
    print("ElasticNet Regression")
    print("Training data RMSE", np.sqrt(mean_squared_error(y_train, pred_train_enet)))
    print("Training data R-squared", r2_score(y_train, pred_train_enet))

    pred_test_enet = model_enet.predict(X_test)
    print("Test data RMSE", np.sqrt(mean_squared_error(y_test, pred_test_enet)))
    print("Test data R-squared", r2_score(y_test, pred_test_enet))

    # result:
    # Linear Regression
    # Training data RMSE 0.4251719568009885
    # Training data R-squared 0.2700808244261864
    # Test data RMSE 0.42610602753210913
    # Test data R-squared 0.26915007582783257

    # Random Forest Regression
    # Training data RMSE 0.1421879483464298
    # Training data R-squared: 0.9183659910371024
    # Test data RMSE 0.37460969170646413
    # Test data R-squared 0.43512693183904405

    # ElasticNet Regression
    # Training data RMSE 0.4788829492105223
    # Training data R-squared 0.07401426251364907
    # Test data RMSE 0.4798343408158256
    # Test data R-squared 0.07322245271238681

    # By comparing RMSE and R-squared, the best performing model is Random Forest with the highest
    # R squared and least RMSE
    # The Linear Regression model is with a better R-squared and RMSE values than the ElasticNet Regression model
    # the ElasticNet Regression model is performing the worst.


def main():
    if not os.path.exists(rootpath + "/plots"):
        os.makedirs(rootpath + "/plots")

    appName = "PySpark-MariaDB baseball"
    master = "local"
    # Create Spark session
    spark = SparkSession.builder.appName(appName).master(master).getOrCreate()

    database = "baseball"
    user = "root"
    password = ""
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"
    sql = "select * from baseball.baseball_stats"

    # Create a data frame by reading data from MariaDB
    baseball_spark_df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .option("query", sql)
        .load()
    )

    baseball_spark_df.persist(StorageLevel.DISK_ONLY)

    # convert spark Df to pandas df
    baseball_df = baseball_spark_df.toPandas()
    baseball_df.dropna()

    response_col = "HomeTeamWins"
    # filter out response with null
    baseball_df = baseball_df[baseball_df[response_col] != ""]

    predictor_cols = [x for x in baseball_df.columns if x != response_col]

    resp_type = determin_type(baseball_df, response_col)

    pr_df = feature_analy_plot_ranking(
        baseball_df, predictor_cols, response_col, resp_type
    )

    (cont_cont_corr_df, cont_cont_cm_plot, cont_cont_mwr_df) = feature_analy_Correl_BF(
        baseball_df, predictor_cols, response_col, resp_type
    )
    create_report(
        pr_df,
        cont_cont_corr_df,
        cont_cont_cm_plot,
        cont_cont_mwr_df,
        predictor_cols,
        response_col,
    )

    # model
    model_compare(baseball_df, predictor_cols, response_col, resp_type)


if __name__ == "__main__":
    sys.exit(main())

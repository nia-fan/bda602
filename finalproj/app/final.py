import sys, os
import mariadb
import numpy as np
import pandas as pd
from plotly import express as px
from plotly import figure_factory as ff
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import statsmodels.api
import statsmodels.formula.api as smf
import webbrowser
import calculate_correlation as calculate_correlation
import datetime


rootpath = os.path.dirname(sys.argv[0])
urlpath = "%s/plots/" % (rootpath)

def determin_type(df, col):
    col_type = df[col].dtype
    if col_type == "object" or col_type== "bool":
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
            file="%s/plots/cont_resp_cont_pre_%s_scatter_plot.html" % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )

        data_dic["plot"] = "<a target='_blank' href='{0}/cont_resp_cont_pre_{1}_scatter_plot.html'>Plot</a>".format(urlpath, pred_col)
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
            yaxis_title="Predictor %s" % pred_col
        )
        # fig_violin.show()
        fig_violin.write_html(
            file="%s/plots/cat_resp_cont_pre_%s_violin_plot.html" % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )

        plot_url = "<a target='_blank' href='{0}/cat_resp_cont_pre_{1}_violin_plot.html'>Violin Plot</a>".format(urlpath, pred_col)

        # distribution plot
        hlist  = df[df[response_col] == "H"][pred_col].values
        alist  = df[df[response_col] == "A"][pred_col].values
        hist_data = [hlist,alist]
       
        fig_dist = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
        fig_dist.update_layout(
            title="Continuous Predictor by Categorical Response",
            xaxis_title="Predictor %s" % pred_col,
            yaxis_title="Distribution"
        )
        # fig_dist.show()
        fig_dist.write_html(
            file="%s/plots/cat_resp_cont_pre_%s_dist_plot.html" % (rootpath, pred_col),
            include_plotlyjs="cdn",
        )
        plot_url = plot_url +  ",&nbsp;&nbsp;<a target='_blank' href='{0}/cat_resp_cont_pre_{1}_dist_plot.html'>Hist Plot</a>".format(urlpath, pred_col)
        data_dic["plot"] = plot_url

def calculate_stats(pred_col, data_dic, df, res_col, resp_type):
    res_value = df[res_col]
    pred_value = df[pred_col]
    if resp_type == "cont":
        x = pred_value
        #add constant to predictor variables
        x = statsmodels.api.add_constant(x)

        #fit linear regression model
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
            yaxis_title="response"
        )
        # fig.show()
        fig.write_html(
            file="%s/plots/linear_regres_%s_plot.html" % (rootpath, res_col),
            include_plotlyjs="cdn",
        )

        data_dic["t-score"] = "{0}, <a target='_blank' href='{1}/linear_regres_{2}_plot.html'>Plot</a>".format(t_value, urlpath, res_col)
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
            yaxis_title="response"
        )
        # fig2.show()
        fig2.write_html(
            file="%s/plots/scatter_%s_plot.html" % (rootpath, res_col),
            include_plotlyjs="cdn",
        )

        data_dic["t-score"] = "{0}, <a target='_blank' href='{1}/scatter_{2}_plot.html'>Plot</a>".format(t_value2, urlpath, res_col)
        data_dic["p-value"] = p_value2

def cal_MWR(pred_col, res_col, data_dic, df):
    df_filter = df.filter(items=[pred_col, res_col])
    
    resp_count = df_filter[res_col].count()
    res_value = df_filter[res_col]
    pred_value = df_filter[pred_col]

    max = np.max(pred_value)
    min = np.min(pred_value)
    bin_size = 10
    bin_width = abs(max - min)/bin_size
    bin_centerValues = np.arange(
        start=min + (bin_width / 2), 
        stop=max, 
        step=bin_width
    )
    bins = np.linspace(min, max, bin_size)

    df_mean = df_filter.groupby(pd.cut(df_filter[pred_col], bins)).mean()
    df_mwr = df_mean[[res_col]].copy()
    df_mwr_w = df_mean[[res_col]].copy()
    mwr_uw = ((df_mwr[res_col] - res_value.mean())**2).sum() / len(bins)
    data_dic["MWR Unweighted"] = mwr_uw

    df_bin_wt = df_filter.groupby(pd.cut(df_filter[pred_col], bins)).count()
    df_bin_wt[res_col] = df_bin_wt[res_col] / resp_count
    data_dic["MWR Weighted"] = (((df_mwr_w[res_col] - res_value.mean())**2) *df_bin_wt[res_col]).sum()
   
    res_mean_lst = [] 
    for i in range(10): 
        res_mean_lst.append(res_value.mean())

    bin_width_lst = [] 
    for i in range(10): 
        bin_width_lst.append(bin_width)

    fig_mwr = go.Figure()
    fig_mwr = make_subplots(specs=[[{"secondary_y": True}]])

    fig_mwr.add_trace(
        go.Bar(x=bin_centerValues, y=df_bin_wt[pred_col], width=bin_width_lst, name="Population"),
        secondary_y=False,
    )
    fig_mwr.add_trace(go.Scatter(x=bin_centerValues, y=res_mean_lst, mode="lines", line_color="green", name="μpop"),secondary_y=True)
    # fig_mwr.add_trace(go.Scatter(x=bins, y=(df_mean[pred_col] - res_value.mean()), mode="lines", line_color="red", name="μi−μpop"), secondary_y=True)
    fig_mwr.add_trace(go.Scatter(x=bin_centerValues, y=(df_mean[pred_col]), mode="lines", line_color="red", name="μi"), secondary_y=True)
    
    # Set titles
    fig_mwr.update_yaxes(title_text="Response", secondary_y=False)
    fig_mwr.update_yaxes(title_text="Population", secondary_y=True)

    fig_mwr.update_layout(
        title="Binned Difference with Mean of Response vs Bin Unweighted (%s)" % pred_col,
        barmode='group',
        bargap=0.30,bargroupgap=0.0)

    fig_mwr.add_annotation(dict(x=0.5,
                                y=-0.06,
                                showarrow=False,
                                text="Predictor Bin",
                                xref="paper",
                                yref="paper"))
    
    # fig_mwr.show()
    fig_mwr.write_html(
        file="%s/plots/MWR_unw_%s_plot.html" % (rootpath, pred_col),
        include_plotlyjs="cdn",
    )

    data_dic["MWR Unweighted"] = "{0}, <a target='_blank' href='{1}/MWR_unw_{2}_plot.html'>Plot</a>".format(mwr_uw, urlpath, pred_col)

def random_forest_ranking(df2, pred_cols, response_col):
    df2["local_date"] = pd.to_datetime(df2["local_date"])
    df2 = df2.set_index(df2["local_date"])
    df2 = df2.sort_index()
    split_date = datetime.datetime(2010,3,1)
    df_training = df2.loc[df2["local_date"] <= split_date]
    df_test = df2.loc[df2["local_date"] > split_date]
    X_train = df_training.filter(items=pred_cols)
    X_test = df_test.filter(items=pred_cols)
    y_train = df_training[response_col]
    y_test = df_test[response_col]

    rfc = RandomForestClassifier(n_estimators=500, random_state=1)
    rfc.fit(X_train, y_train)

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
        cal_MWR(column, response_col, data_dic,df2)

        results_data.append(data_dic)
    
    rf_list = random_forest_ranking(df2, predictor_cols, response_col)
    index = 0
    for item in results_data:
        item["RF VarImp"] = rf_list[index]
        index +=1
    
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
                pair_name = pred_cont_df.columns[index] + " and " + pred_cont_df.columns[index2]
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
                    yaxis_title=pred_cont_df.columns[index2]
                )
                # fig.show()
                fig.write_html(
                    file="%s/plots/%s_linear_regression.html" % (rootpath, pair_name.replace(" ", "_")),
                    include_plotlyjs="cdn",
                )

                cont_cont_dic["Linear Regression Plot"] = pair_name.replace(" ", "_") + "_linear_regression"
                cont_cont_list.append(cont_cont_dic)

    
    cont_cont_list = sorted(cont_cont_list, key=lambda x: x["Absolute Value of Correlation"], reverse=True)
    cont_cont_df = pd.DataFrame.from_dict(cont_cont_list)
    # create html link
    if "Linear Regression Plot" in cont_cont_df.columns:
        cont_cont_df["Linear Regression Plot"] = cont_cont_df["Linear Regression Plot"].apply(lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(urlpath, x, x))

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
                df_mean = df_filter.groupby([pd.cut(df_filter[c1], bin_num), pd.cut(df_filter[c2], bin_num)]).mean().unstack()
                df_mean_weight = df_mean.copy()

                max = np.max(df_filter[c1])
                min = np.min(df_filter[c1])
                y = np.linspace(min, max , 6)

                max = np.max(df_filter[c2])
                min = np.min(df_filter[c2])
                x = np.linspace(min, max , 6)

                # bin count
                df__bin_count = df_filter.groupby([pd.cut(df_filter[c1], bin_num), pd.cut(df_filter[c2], bin_num)]).count().unstack()
                df_bin_wt = df__bin_count.copy()
                # population proportion
                df__bin_count[response_col] = df__bin_count[response_col] / resp_count

                cont_wmr_dic["Predictor 1"] = c1
                cont_wmr_dic["Predictor 2"] = c2

                df_mean_new = df_mean[[response_col]].copy()
                # calculate mwr
                df_mean_new[response_col] = (df_mean[response_col] - resp_mean) **2
                cont_wmr_dic["Difference of Mean Response"] = df_mean_new[response_col].sum().sum() / (len(x) * len(y))
               
                df_mean_mwr_w = df_mean.copy()
                # weight
                df_bin_wt[response_col] = df_bin_wt[response_col] / resp_count
                df_mean_mwr_w[response_col] = ((df_mean_mwr_w[response_col] - resp_mean) **2) * df_bin_wt[response_col]
                cont_wmr_dic["Weighted Difference of Mean Response"] = df_mean_mwr_w[response_col].sum().sum()
              
                # unweighted plot
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x = x,
                        y = y,
                        z = df_mean[response_col],
                        text=df_mean[response_col],
                        texttemplate='%{text:.3f}',
                        colorscale='rdylgn',
                        customdata = df__bin_count[response_col]
                        )
                    )
                fig.update_traces(
                    hovertemplate="<br>".join([
                        "X: %{x}",
                        "Y: %{y}",
                        "Z: %{z}",
                        "Population Proportion: %{customdata:.3f}"
                    ])
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c2, c1),
                    xaxis_title = c2,
                    yaxis_title = c1

                )
                # fig.show()
                fig.write_html(
                        file="%s/plots/%s_bin_uw_plot.html" % (rootpath, pair_name.replace(" ", "_")),
                        include_plotlyjs="cdn",
                    )
                
                cont_wmr_dic["Bin Plot"] = pair_name.replace(" ", "_") + "_bin_uw_plot"
               
                # weighted plot
                df_mean_weight[response_col] = (df_mean_weight[response_col] - resp_mean)
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(
                        x = x,
                        y = y,
                        z = df_mean_weight[response_col],
                        text=df_mean_weight[response_col],
                        texttemplate='%{text:.3f}',
                        colorscale='blugrn',
                        customdata = df__bin_count[response_col]
                        )
                    )
                fig.update_traces(
                    hovertemplate="<br>".join([
                        "X: %{x}",
                        "Y: %{y}",
                        "Z: %{z}",
                        "Population Proportion: %{customdata:.3f}"
                    ])
                )
                fig.update_layout(
                    title="%s and %s Bin Average Response" % (c2, c1),
                    xaxis_title = c2,
                    yaxis_title = c1

                )
                # fig.show()
                fig.write_html(
                        file="%s/plots/%s_dwm_of_resp_residual.html" % (rootpath, pair_name.replace(" ", "_")),
                        include_plotlyjs="cdn",
                    )
                
                cont_wmr_dic["Residual Plot"] = pair_name.replace(" ", "_") + "_dwm_of_resp_residual"
                cont_cont_mwr_list.append(cont_wmr_dic)

   
    cont_cont_mwr_list = sorted(cont_cont_mwr_list, key=lambda x: x["Weighted Difference of Mean Response"], reverse=True)
    cont_cont_mwru_df = pd.DataFrame.from_dict(cont_cont_mwr_list)
    if "Bin Plot" in cont_cont_mwru_df.columns:
        cont_cont_mwru_df["Bin Plot"] = cont_cont_mwru_df["Bin Plot"].apply(lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(urlpath, x, x))
    if "Residual Plot" in cont_cont_mwru_df.columns:
        cont_cont_mwru_df["Residual Plot"] = cont_cont_mwru_df["Residual Plot"].apply(lambda x: "<a target='_blank' href='{0}/{1}.html'>{2}</a>".format(urlpath, x, x))

    return cont_cont_mwru_df

def feature_analy_Correl_BF(df, predictor_cols, response_col, resp_type):
    pred_cont_df = df.filter(items=predictor_cols)
    cont_cont_corr_df = cal_cont_cont_corre(pred_cont_df)
    # correlation matrix
    cont_cont_corr_matrix = pred_cont_df.corr()   
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = cont_cont_corr_matrix.columns,
            y = cont_cont_corr_matrix.index,
            z = np.array(cont_cont_corr_matrix),
            colorscale='blugrn'
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
    pred_str = '<br />'.join(str(x) for x in predictor_cols)
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
            width="1400" height="1000" frameborder="0" align="middle">
        </iframe>
        </div>
        <div></div>
        <h3 style='text-align: center;'>"Brute Force" Table</h3>
        <div>%s</div>
        <br/><br/>
        </body>
        </html>
        """ % (pr_df.to_html(escape=False, index=False, classes='table table-stripped'), 
            response_col, 
            pred_str,
            df1.to_html(escape=False, index=False, classes='table table-stripped') if df1 is not None and not df1.empty else "",
            cm1,
            mwrdf1.to_html(escape=False, index=False, classes='table table-stripped') if mwrdf1 is not None and not mwrdf1.empty else "")

    f = open(rootpath + '/report.html', 'w')
    f.write(html_template)
    f.close()

    new = 2 # open in a new tab, if possible
    url = "file://%s/report.html" % rootpath
    webbrowser.open(url,new=new)


def model_compare(df, pred_col, response_col, resp_type):
    df2 = df.copy()
    # convert response to numeric
    if resp_type == "cat":
        le = LabelEncoder()
        label_encd = le.fit_transform(df2[response_col])
        df2[response_col] = label_encd

    # since the game date is from 2007-2-28 to 2012-6-28
    # we set the split date as 2011-3-1, all data before are training data, after are test data
    df2["local_date"] = pd.to_datetime(df["local_date"])
    df2 = df2.set_index(df2["local_date"])
    df2 = df2.sort_index()
    split_date = datetime.datetime(2010,3,1)
    # split_date = datetime.datetime(2009,3,1)
    df_training = df2.loc[df2["local_date"] <= split_date]
    df_test = df2.loc[df2["local_date"] > split_date]
    X_train = df_training.filter(items=pred_col)
    X_test = df_test.filter(items=pred_col)
    y_train = df_training[response_col]
    y_test = df_test[response_col]

    # Classifiers
    # 1. logistic regression model
    logistic_model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test,y_pred)
    print("logistic regression model...")
    print("Accuracy: ",accuracy)
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))
    print('F1 Score: ', metrics.f1_score(y_test, y_pred))
    print("Matthew's correlation:", metrics.matthews_corrcoef(y_test, y_pred))

    # 2. random forest classifier
    rfc_model = RandomForestClassifier()
    rfc_model.fit(X_train,y_train)
    rfc_predictions = rfc_model.predict(X_test)
    print("Random Forest Classifier...")
    print("Accuracy: ", metrics.accuracy_score(rfc_predictions, y_test))
    print("Precision: ", metrics.precision_score(y_test, rfc_predictions))
    print("Recall: ", metrics.recall_score(y_test, rfc_predictions))
    print('F1 Score: ', metrics.f1_score(y_test, rfc_predictions))
    print("Matthew's correlation:", metrics.matthews_corrcoef(y_test, rfc_predictions))

    # 3. Support Vector Classifier
    SVC_model = SVC()
    SVC_model.fit(X_train, y_train)
    SVC_prediction = SVC_model.predict(X_test)
    print("Support Vector Classifier...")
    print("Accuracy: ", metrics.accuracy_score(SVC_prediction, y_test))
    print("Precision: ", metrics.precision_score(y_test, SVC_prediction))
    print("Recall: ", metrics.recall_score(y_test, SVC_prediction))
    print('F1 Score: ', metrics.f1_score(y_test, SVC_prediction))
    print("Matthew's correlation:", metrics.matthews_corrcoef(y_test, SVC_prediction))

    # 4. Classification Task with Naive Bayes
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)
    gnb_prediction = gnb_model.predict(X_test)
    # Evaluate label (subsets) accuracy:
    print("Naive Bayes Classifier...")
    print("Accuracy: ", metrics.accuracy_score(gnb_prediction,y_test))
    print("Precision: ", metrics.precision_score(y_test, gnb_prediction))
    print("Recall: ", metrics.recall_score(y_test, gnb_prediction))
    print('F1 Score: ', metrics.f1_score(y_test, gnb_prediction))
    print("Matthew's correlation:", metrics.matthews_corrcoef(y_test, gnb_prediction))
    
def turn_features(df, pred_cols, response_col, resp_type):
    df2 = df.copy()
    # convert response to numeric
    if resp_type == "cat":
        le = LabelEncoder()
        label_encd = le.fit_transform(df2[response_col])
        df2[response_col] = label_encd

    rf_ranking= random_forest_ranking(df2, pred_cols, response_col)

    index = 0
    ranking = []
    for col in pred_cols:
        data = {}
        data["col"] = col
        data["ranking"] = rf_ranking[index]
        ranking.append(data)
        index +=1

    ranking.sort(key = lambda x : x["ranking"],reverse=True)
    for r in ranking:
       print("%s: %s" % (r["col"], r["ranking"]))

def main():
    if not os.path.exists(rootpath + "/plots"):
        os.makedirs(rootpath + "/plots")

    database = "baseball"
    user = "root"
    password = "xxxxx"
    # server = "localhost"
    server = "mariaDB"   #for docker
    port = 3306
    
    sql = "select * from baseball.baseball_game_stats"

    # Connect to MariaDB Platform
    try:
        conn = mariadb.connect(
            user=user,
            password=password,
            host=server,
            port=port,
            database=database

        )
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB Platform: {e}")
        sys.exit(1)
    
    # Get Cursor
    mycursor = conn.cursor()
    mycursor.execute(sql)
    table_rows = mycursor .fetchall()
    columns = [col[0] for col in mycursor.description]

    baseball_df = pd.DataFrame(table_rows, columns=columns)
    baseball_df = baseball_df.drop(columns=['away_finalScore', 'home_finalScore'])
    baseball_df=baseball_df.dropna().reset_index(drop=True)
    # print(baseball_df.head())

    response_col = "HomeTeamWins"
    # filter out response with T
    baseball_df = baseball_df[baseball_df[response_col] != "T"]
    resp_type = determin_type(baseball_df, response_col)

    # tune the data
    # 1. drop following 4 featuers with low random forest ranking
    # baseball_df = baseball_df.drop(columns=["away_HomeRun_rolling","home_HomeRun_rolling","away_startingPitcher_rolling","home_startingPitcher_rolling"])
    # # 2. drop foloowing features with low random forest ranking
    # baseball_df = baseball_df.drop(columns=["away_Strikeout_rolling","home_Strikeout_rolling","away_FieldError_rolling","home_FieldError_rolling"])
    # # 3. drop foloowing features with low random forest ranking
    # baseball_df = baseball_df.drop(columns=["away_TOB_rolling","home_TOB_rolling","away_TB_rolling","away_WHIP_rolling","away_OBP_rolling","home_TB_rolling"])
    # # 4. drop features with negative t-score
    # baseball_df = baseball_df.drop(columns=["home_WHIP_rolling","home_wOBA_rolling","home_HR_H_rolling","away_RDIFF_rolling","away_HR_H_rolling","home_K9_rolling"])

    # 5 drop features based on correlation > 90% and lower random forest ranking
    baseball_df= baseball_df.drop(columns=["away_WHIP_rolling","home_WHIP_rolling","away_HomeRun_rolling","home_HomeRun_rolling","away_WHIP_rolling","home_WHIP_rolling",
                                        "home_Strikeout_rolling","away_Strikeout_rolling","home_OBP_WHIP_Diff","away_OBP_WHIP_Diff", "away_TB_rolling","home_TOB_rolling",
                                        "home_wOBA_rolling","away_wOBA_rolling","away_SLG_rolling","home_K9_rolling","away_GOAO_rolling","away_K9_rolling",
                                        "home_SLG_rolling","away_HR_H_rolling","home_HR_H_rolling"])

    # 6 drop features based on correlation > 82% and lower random forest ranking
    baseball_df= baseball_df.drop(columns=["home_OBP_rolling","away_OBP_rolling","home_HR_HomeRun_Diff","away_TB_TOB_Diff","home_TB_TOB_Diff","away_HR_HomeRun_Diff",
                                            "home_wOBA_SLG_Diff","away_wOBA_SLG_Diff","home_startingPitcher_rolling","away_startingPitcher_rolling"])


    predictor_cols = [x for x in baseball_df.columns if (x != response_col and x!="game_id" and x!="local_date")]

    pr_df = feature_analy_plot_ranking(baseball_df, predictor_cols, response_col, resp_type)
    (cont_cont_corr_df, cont_cont_cm_plot, cont_cont_mwr_df) = feature_analy_Correl_BF(baseball_df, predictor_cols, response_col, resp_type)
    create_report(pr_df, cont_cont_corr_df, cont_cont_cm_plot, cont_cont_mwr_df, predictor_cols, response_col)

    # this is for testing
    # turn_features(baseball_df, predictor_cols, response_col, resp_type)
    # model
    model_compare (baseball_df, predictor_cols, response_col, resp_type)

if __name__ == "__main__":
    sys.exit(main())
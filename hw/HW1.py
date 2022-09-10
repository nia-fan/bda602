import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

def main():

    # 1. load data in df
    in_file_path = "~/bda602/hw/iris.data"

    # read data based on column names

    df = pd.read_csv(
        in_file_path,
        names=[
            "Sepal_Length",
            "Sepal_Width",
            "Petal_Length",
            "Petal_Width",
            "Iris_Class",
        ],
    )

    # 2. calculate mean, min, max, quartiles
    # get np array from df excluding the Iris_Class field
    np_arr = df.loc[:, "Sepal_Length":"Petal_Width"]
    mean = np.mean(np_arr, axis=0)
    print("mean...")
    print(mean)
    max = np.max(np_arr, axis=0)
    print("max...")
    print(max)
    min = np.min(np_arr, axis=0)
    print("min...")
    print(min)
    print("quartiles...")
    print(np.quantile(np_arr, 0.25, axis=0))
    print(np.quantile(np_arr, 0.50, axis=0))
    print(np.quantile(np_arr, 0.75, axis=0))

    # 3. plot
    # scatter plot with all data
    fig = px.scatter(df, x="Sepal_Width", y="Sepal_Length", color="Iris_Class")
    fig.show()

    # violin plot Iris-setosa vs Iris-versicolor
    fig = px.violin(
        df.query('Iris_Class=="Iris-setosa" or Iris_Class=="Iris-versicolor"'),
        y="Sepal_Length",
        x="Iris_Class",
        color="Iris_Class",
        box=True,  # draw box plot inside the violin
        points="all",
    )
    fig.show()

    # violin plot Iris-setosa vs Iris-virginica
    fig = px.violin(
        df.query('Iris_Class=="Iris-setosa" or Iris_Class=="Iris-virginica"'),
        y="Sepal_Length",
        x="Iris_Class",
        color="Iris_Class",
        box=True,  # draw box plot inside the violin
        points="all",
    )
    fig.show()

    # violin plot Iris-virginica vs Iris-versicolor
    fig = px.violin(
        df.query('Iris_Class=="Iris-virginica" or Iris_Class=="Iris-versicolor"'),
        y="Sepal_Length",
        x="Iris_Class",
        color="Iris_Class",
        box=True,
        points="all",
    )
    fig.show()

    # scatter matrix for entire data set
    fig = px.scatter_matrix(
        df,
        dimensions=["Sepal_Width", "Sepal_Length", "Petal_Width", "Petal_Length"],
        color="Iris_Class",
    )
    fig.show()

    # 4. Analyze and build models - Use scikit-learn
    # convert Iris_Class into numbers
    le = LabelEncoder()
    df["Iris_Class"] = le.fit_transform(df["Iris_Class"])

    # divide DataFrame into features (X) and target (y)
    X = df.drop("Iris_Class", axis=1)
    y = df["Iris_Class"]

    # split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=50
    )

    # fit and transform
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)

    scaler.fit(X_test)
    scaler.transform(X_test)

    # build Random Forest Classifier
    rfc = RandomForestClassifier(random_state=50)
    rfc.fit(X_train, y_train)

    # support vector machine SVC classifier
    clf = svm.SVC()
    clf.fit(X_train, y_train)

    # 5.Wrap the steps into a pipeline
    pipe_random_f = Pipeline(
        [("scaler", StandardScaler()), ("rfc", RandomForestClassifier(random_state=50))]
    )
    pipe_random_f.fit(X_train, y_train)
    Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rfc", RandomForestClassifier(random_state=50)),
        ]
    )
    pipe_svc = Pipeline([("scaler", StandardScaler()), ("clf", svm.SVC())])
    pipe_svc.fit(X_train, y_train)
    Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", svm.SVC()),
        ]
    )
    print("Accuracy of random forest:", (pipe_random_f.score(X_test, y_test)))
    print("Accuracy of support vector machine:", (pipe_svc.score(X_test, y_test)))


if __name__ == "__main__":
    main()



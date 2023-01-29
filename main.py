import pandas as pd
from NB_Me import *


if __name__ == '__main__':
    df_train = pd.read_csv("train_dataset_nb.csv")

    nb_classifier = NBClfMest()
    nb_classifier.train(df_train)

    df_test = pd.read_csv("test_dataset_nb.csv")

    X_test = df_test.iloc[:, :-1].values.tolist()
    y_test = df_test.iloc[:, -1].tolist()

    y_pred = nb_classifier.predict(X_test)

    accuracy = nb_classifier.accuracy(y_test, y_pred)

    print("Accuracy ==> {}".format(accuracy))
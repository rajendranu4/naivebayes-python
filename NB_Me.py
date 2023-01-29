import pandas as pd
import numpy as np


class NBClfMest:
    def __init__(self):
        self.model_name = "NaiveBayes-M_Estimate"

    def get_string(self, name):

        # this is just an utility method to return a string in format P(<<name>>)
        string_name = None
        if (isinstance(name, int)):
            string_name = 'P(' + str(name) + ')'
        else:
            string_name = 'P(' + name + ')'
        return string_name

    def train(self, df):

        ''' initializing the class attributes such as likelihood probability, prior probability, feature_names
            and class_variables from dataset'''
        self.class_var = df.columns[-1]
        self.columns = df.columns
        self.classes = df[self.class_var].unique().tolist()
        self.likelihood = {}
        self.prior = pd.DataFrame(columns=[self.class_var], index=[df[self.class_var].unique()])

        # variables for preparing likelihood-probability dataframe/table
        likelihood_index = df[self.class_var].unique().tolist()
        likelihood_index_count = df[self.class_var].nunique()

        likelihood_index_prob = [self.get_string(index) for index in likelihood_index]
        likelihood_index.extend(likelihood_index_prob)

        # preparing the structure of the likelihood-probability dataframe
        # indices are the class values and columns are the values that a feature can have
        for i in range(0, (len(df.columns) - 1)):
            likelihood_param = df.columns[i]
            cols = np.sort(df[likelihood_param].unique())
            self.likelihood[likelihood_param] = pd.DataFrame(columns=cols, index=likelihood_index)

        # frequencies / number of occurences of a feature_value corresponding to a class variable is stored
        for feature in self.likelihood:
            frequency_group = df.groupby([self.class_var, feature]).size()

            for index, frequency in frequency_group.items():
                self.likelihood[feature].loc[index[0], index[1]] = frequency

        # likelihood-probability P(X=x | class) is then calculated for each feature_value
        for feature in self.likelihood:
            for feature_value in self.likelihood[feature]:
                # number of instances of that particular feature_value is counted from likelihood table
                count_instances = self.likelihood[feature][feature_value].sum()

                # m-estimate version of NB is used here with below parameters
                # m --> number of unique values that the feature has
                # p --> 1 / number of unique values that the feature has
                m = df[feature].nunique()
                p = (1 / df[feature].nunique())

                for class_v in likelihood_index:
                    class_v_prob = self.get_string(class_v)
                    Nc = self.likelihood[feature][feature_value][class_v]
                    N = self.likelihood[feature].loc[class_v].sum()

                    # likelihood probability is stored in the table
                    self.likelihood[feature][feature_value][class_v_prob] = ((Nc + (m * p)) / (N + m))

        '''for lh in self.likelihood:
            print(lh)
            print(self.likelihood[lh])
            print("\n")'''

        # variables for preparing prior-probability table
        prior_classprob = df.groupby(self.class_var).size()
        total_instances = df.shape[0]

        # prior probability is calculated for each of the class labels
        # count of a class label is divided by total instances gives P(class label)
        for index, frequency in prior_classprob.items():
            self.prior.loc[index, self.class_var] = (frequency / total_instances)

        # print(self.prior)

    def predict(self, X_test):

        # initializing the prediction class labels list
        y_pred = [None] * len(X_test)

        # each of the instance from X_test goes through testing
        for i in range(len(X_test)):
            max_prob_class = None
            max_prob = 0

            # probability for each class label is calculated and the class label with maximum probability for a test
            # instance is considered as the classified/predicted class label for that instance
            for _class in self.classes:
                fv = 0

                # probability is calculated as P(class)*P(X=x1|class)*P(X=x2|class)*..*P(X=xn|class)
                prob = self.prior.loc[_class, self.class_var]

                for feature in self.columns[:-1]:
                    prob *= self.likelihood[feature][X_test[i][fv]][self.get_string(_class)]
                    fv = fv + 1

                if float(prob) > max_prob:
                    max_prob_class = _class
                    max_prob = float(prob)

            y_pred[i] = max_prob_class

        return y_pred

    def accuracy(self, original, predicted):
        correct_pred_count = 0

        # calculating the number of correctly predicted class labels
        for i in range(len(predicted)):
            if original[i] == predicted[i]:
                correct_pred_count += 1

        # print("\nTotal correct predictions ==> {} / {}".format(correct_pred_count, len(original)))
        return correct_pred_count / len(original)
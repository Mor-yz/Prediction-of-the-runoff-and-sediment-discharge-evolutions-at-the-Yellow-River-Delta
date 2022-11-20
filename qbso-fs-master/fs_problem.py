from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.preprocessing import StandardScaler
from solution import Solution
from NS import ns
from sklearn.metrics import r2_score


class FsProblem:
    def __init__(self, typeOfAlgo, data, qlearn, classifier=Ridge(alpha=1)):
        self.data = data
        self.nb_attribs = len(
            self.data.columns) - 1  # The number of features is the size of the dataset - the 1 column of labels
        self.outPuts = self.data.iloc[:, self.nb_attribs]  # We initilize the labels from the last column of the dataset
        self.ql = qlearn
        self.classifier = classifier
        self.typeOfAlgo = typeOfAlgo

    def evaluate2(self, solution):
        sol_list = Solution.sol_to_list(solution)
        if len(sol_list) == 0:
            return 0

        df = self.data.iloc[:, sol_list]
        array = df.values
        X = array[:, 0:self.nb_attribs]
        Y = self.outPuts
        train_X, test_X, train_y, test_y = train_test_split(X, Y,
                                                            random_state=0,
                                                            test_size=0.25
                                                            )

        self.classifier.fit(train_X, train_y)
        predict = self.classifier.predict(test_X)
        return r2_score(test_y, predict)
        # return metrics.accuracy_score(predict, test_y)
        # return - abs(metrics.mean_squared_error(predict, test_y))

    def evaluate(self, solution):
        sol_list = Solution.sol_to_list(solution)
        if len(sol_list) == 0:
            return 0

        df = self.data.iloc[:, sol_list]  # For this function you need to put the indexes of features you picked
        array = df.values
        X = array[:, 0:self.nb_attribs]
        Y = self.outPuts
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)  # Cross validation function
        results = cross_val_score(self.classifier, X, Y, cv=cv, scoring='accuracy')
        # print("\n[Cross validation results]\n{0}".format(results))
        return results.mean()

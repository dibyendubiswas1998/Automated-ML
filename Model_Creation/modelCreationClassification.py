from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from Application_Log.logger import App_Logger


class Classification_Model_Finder:
    """
        This class shall be used for create the Models for classification

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """

    def __init__(self, file_path):
        self.file_path = file_path  # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Model_Creation_Logs.txt"
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def CreateDecisionTreeClassifier(self, x_train, y_train):
        """
            Method Name: CreateDecisionTreeClassifier
            Description: This method helps to create decision tree model after hyperparameter tuning.

            Output: model with the best parameter.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "We use DecisionTree Classifier")
            self.x_train = x_train
            self.y_train = y_train
            self.clf = DecisionTreeClassifier()  # declare base decision tree model.
            self.set_params = {
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 42, 1),
                'min_samples_leaf': range(1, 10, 1),
                'min_samples_split': range(2, 10, 1),
                'splitter': ['best', 'random']
            }
            self.grid_search = GridSearchCV(estimator=self.clf, param_grid=self.set_params, cv=7,
                                            n_jobs=-1)  # apply GridSearch to find the best parameter
            self.grid_search.fit(self.x_train, self.y_train)
            # get the parameters after applying GridSearch algorithm:
            self.criterion = self.grid_search.best_params_['criterion']
            self.max_depth = self.grid_search.best_params_['max_depth']
            self.min_samples_leaf = self.grid_search.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid_search.best_params_['min_samples_split']
            self.splitter = self.grid_search.best_params_['splitter']
            self.logger_object.log(self.file,
                                   f"Get the best parameters after hyperparameter tuning {self.grid_search.best_params_}")

            # train the model with the best parameters.
            self.clf = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                              max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf)
            self.logger_object.log(self.file, f"train the model with best parameters {self.grid_search.best_params_}")
            self.file.close()
            return self.clf  # return the model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def CreateRandomForestClassifie(self, x_train, y_train):
        """
            Method Name: CreateRandomForestClassifie
            Description: This method helps to create model after hyperparameter tuning.

            Output: model with the best parameter.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "We use RandomForest Classifier")
            self.x_train = x_train
            self.y_train = y_train
            self.set_params = {"n_estimators": [10, 50, 100, 130, 150], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}  # set the hyperparameter

            self.clf = RandomForestClassifier()  # declare the base model
            self.grid_search = GridSearchCV(estimator=self.clf, param_grid=self.set_params, cv=7,
                                            verbose=3)  # apply GridSearch to find the best parameter
            self.grid_search.fit(self.x_train, self.y_train)
            # get the parameters after applying GridSearch algorithm:
            self.criterion = self.grid_search.best_params_['criterion']
            self.max_depth = self.grid_search.best_params_['max_depth']
            self.max_features = self.grid_search.best_params_['max_features']
            self.n_estimators = self.grid_search.best_params_['n_estimators']
            self.logger_object.log(self.file,
                                   f"Get the best parameters after hyperparameter tuning {self.grid_search.best_params_}")

            # train the model with best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth,
                                              max_features=self.max_features)
            self.logger_object.log(self.file, f"Trained the model with best parameter {self.grid_search.best_params_}")
            self.file.close()
            return self.clf  # return model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def CreateXGBoostClassifier(self, x_train, y_train):
        """
            Method Name: CreateXGBoostClassifier
            Description: This method helps to create XGboost model after hyperparameter tuning.

            Output: model with the best parameter.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "We use XGBoost Classifier")
            self.x_train = x_train
            self.y_train = y_train
            self.clf = XGBClassifier(objective='binary:logistic')
            self.set_params = {
                'learning_rate': [0.5, 0.1, 0.01, 0.001],  # set the parameter
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 150, 200]
            }
            self.grid_search = GridSearchCV(estimator=self.clf, param_grid=self.set_params, cv=5,
                                            verbose=3)  # apply GridSearch to find the best parameter
            self.grid_search.fit(x_train, y_train)
            # get the parameters after applying GridSearch algorithm:
            self.learning_rate = self.grid_search.best_params_['learning_rate']
            self.max_depth = self.grid_search.best_params_['max_depth']
            self.n_estimators = self.grid_search.best_params_['n_estimators']
            self.logger_object.log(self.file,
                                   f"Get the best parameters after hyperparameter tuning {self.grid_search.best_params_}")

            # train the model with the best parameters:
            self.clf = XGBClassifier(self.learning_rate, self.max_depth, self.n_estimators)
            self.logger_object.log(self.file, f"Trained the model with best parameters {self.grid_search.best_params_}")
            self.file.close()
            return self.clf

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def CreateEnsembleTechniquesDecisionTee(self, x_train, y_train):
        """
            Method Name: CreateEnsembleTechniquesDecisionTee
            Description: This method helps to create model using ensemble techniques
                         where base model is Decision Tree.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "We use Ensemble approach, where base model is Decision Trees")
            self.x_train = x_train
            self.y_train = y_train
            self.clf = DecisionTreeClassifier()  # declare base decision tree model.
            self.set_params = {
                'criterion': ['gini', 'entropy'],
                'max_depth': range(2, 42, 1),
                'min_samples_leaf': range(1, 10, 1),
                'min_samples_split': range(2, 10, 1),
                'splitter': ['best', 'random']
            }
            self.grid_search = GridSearchCV(estimator=self.clf, param_grid=self.set_params, cv=7,
                                            n_jobs=-1)  # apply GridSearch to find the best parameter
            self.grid_search.fit(self.x_train, self.y_train)
            # get the parameters after applying GridSearch algorithm:
            self.criterion = self.grid_search.best_params_['criterion']
            self.max_depth = self.grid_search.best_params_['max_depth']
            self.min_samples_leaf = self.grid_search.best_params_['min_samples_leaf']
            self.min_samples_split = self.grid_search.best_params_['min_samples_split']
            self.splitter = self.grid_search.best_params_['splitter']
            self.logger_object.log(self.file,
                                   f"Get the best parameters after hyperparameter tuning {self.grid_search.best_params_}")

            # train the model with the best parameters.
            self.clf = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter,
                                              max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf)
            self.logger_object.log(self.file, f"train the model with best parameters {self.grid_search.best_params_}")
            self.clf = BaggingClassifier(base_estimator=self.clf, n_estimators=10, max_samples=0.5,
                                         bootstrap=True, random_state=101, oob_score=True)
            self.logger_object.log(self.file, f"Trained the model using Ensembel approach")
            self.file.close()
            return self.clf  # return the model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def CreateEnsembleTechniquesKNN(self, x_train, y_train):
        """
            Method Name: CreateEnsembleTechniquesKNN
            Description: This method helps to create model using ensemble techniques
                         where base model is KNN.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "We use Ensemble approach, where base model is KNN")
            self.x_train = x_train
            self.y_train = y_train
            self.clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                            metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                                            weights='uniform')
            self.clf = BaggingClassifier(base_estimator=self.clf, n_estimators=17, max_samples=0.5,
                                         bootstrap=True, random_state=101, oob_score=True)
            self.logger_object.log(self.file, f"Trained the model using Ensembel approach")
            self.file.close()
            return self.clf

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass

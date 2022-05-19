from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from Application_Log.logger import App_Logger


class Classification_Model_Finder:
    """
        This class shall be used for create the Models for classification

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_path):
        self.file_path = file_path   # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Model_Creation_Logs.txt"
        self.logger_object = App_Logger()   # call the App_Logger() to log the details

    def RandomForestClassifie(self, x_train, y_train):
        """
            Method Name: RandomForestClassifie
            Description: This method helps to create model after hyperparameter tuning.

            Output: model with the best parameter.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.x_train = x_train
            self.y_train = y_train
            self.set_params = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}  # set the hyperparameter

            self.clf = RandomForestClassifier()  # declare the base model
            self.grid_search = GridSearchCV(estimator=self.clf, param_grid=self.set_params, cv=7, verbose=3)  # apply GridSearch to find the best parameter
            self.grid_search.fit(self.x_train, self.y_train)
            # get the parameters after applying GridSearch algorithm:
            self.criterion = self.grid_search.best_params_['criterion']
            self.max_depth = self.grid_search.best_params_['max_depth']
            self.max_features = self.grid_search.best_params_['max_features']
            self.n_estimators = self.grid_search.best_params_['n_estimators']
            self.logger_object.log(self.file, f"Get the best parameters after hyperparameter tuning {self.grid_search.best_params_}")

            # train the model with best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
                                              max_features=self.max_features)
            self.logger_object.log(self.file, f"Trained the model with best parameter {self.grid_search.best_params_}")
            self.file.close()
            return self.clf  # return model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


class Regression_Model_Finder:
    """
         This class shall be used for create the Models for regression.

         Written By: Dibyendu Biswas.
         Version: 1.0
         Revisions: None
    """
    def __init__(self, file_path):
        self.file_path = file_path  # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Model_Creation_Logs.txt"
        self.logger_object = App_Logger()  # call the App_Logger() to log the details


    def CreateLinearRegression(self, x_train, y_train):
        """
            Method Name: CreateLinearRegression
            Description: This method helps to create model using Linear regression.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "Use Linear Regression to create the model")
            self.x_train = x_train
            self.y_train = y_train
            self.reg = LinearRegression()   # use LinearRegression algirithm
            self.reg.fit(self.x_train, self.y_train)
            self.logger_object.log(self.file, "Trained the model using LinearRegression algorithm")
            self.file.close()
            return self.reg  # return the model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


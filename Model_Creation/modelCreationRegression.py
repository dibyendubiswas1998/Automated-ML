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
            self.reg = LinearRegression()  # use LinearRegression algirithm
            self.reg.fit(self.x_train, self.y_train)
            self.logger_object.log(self.file, "Trained the model using LinearRegression algorithm")
            self.file.close()
            return self.reg  # return the model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def CreateRidgeRegression(self, x_train, y_train):
        """
            Method Name: CreateRidgeRegression
            Description: This method helps to create model using Ridge Regression.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "Use Ridge Regression to create the model")
            # finding the best parameters using RidgeCV
            self.alpha = np.random.uniform(low=0, high=10, size=(50,))
            self.ridgecv = RidgeCV(alphas=self.alpha, cv=10, normalize=True)
            self.ridgecv.fit(self.x_train, self.y_train)
            self.alpha_ = self.ridgecv.alpha_  # get the alpha value using RidgeCV
            self.logger_object.log(self.file, f"Get the alpha value using RidgeCV {self.alpha_}")
            self.reg = Ridge(alpha=self.alpha_)  # using alpha value try to train the model
            self.reg.fit(self.x_train, self.y_train)
            self.logger_object.log(self.file, "Successfully trained the model")
            self.file.close()
            return self.reg  # return regression model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def CreateLassoRegression(self, x_train, y_train):
        """
            Method Name: CreateLassoRegression
            Description: This method helps to create model using Lasso Regression.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "Use Lasso Regression to create the model")
            self.x_train = x_train
            self.y_train = y_train
            # finding the best parameters using LassoCV
            self.lassocv = LassoCV(alphas=None, cv=10, max_iter=100000, normalize=True)  # tune the hyperparameter.
            self.alpha_ = self.lassocv.alpha_  # get the alpha value
            self.logger_object.log(self.file, f"get the alpha value using LassoCv {self.alpha_}")
            self.reg = Lasso(alpha=self.alpha_)
            self.reg.fit(self.x_train, self.y_train)  # train the model with Lasso Regression
            self.logger_object.log(self.file, "Successfully trained the model using Lasso Regression")
            self.file.close()
            return self.reg  # return the regession model.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def CreateElasticNet(self, x_train, y_train):
        """
            Method Name: CreateElasticNet
            Description: This method helps to create model using ElsticNet.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "Use ElasticNet to create the model")
            self.x_train = x_train
            self.y_train = y_train
            # finding the best parameters using ElasticNetCV
            self.elasticCV = ElasticNetCV(alphas=None, cv=20)
            self.alpha, self.l1_ratio = self.elasticCV.alpha_, self.elasticCV.l1_ratio  # get the alpha & l1_ratio
            self.logger_object.log(self.file, f"Get the alpha {self.alpha} and l1_ration {self.l1_ratio}")
            self.reg = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)  # train the model with best hyperparameter
            self.reg.fit(self.x_train, self.y_train)
            self.logger_object.log(self.file, "Successfully trained the model using ElasticNet")
            self.file.close()
            return self.reg

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


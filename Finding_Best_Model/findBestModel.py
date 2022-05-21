from Application_Log.logger import App_Logger
from Model_Creation.modelCreationRegression import To_Create_Regression_Model as Reg
from Model_Creation.modelCreationClassification import To_Create_Classification_Model as Cls
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, confusion_matrix
import numpy as np
import pandas as pd


class Find_Best_Model:
    """
         This class shall be used for find the best model either classification or regression.
         and it's helps to log the accuracy for different different algirithms.

         Written By: Dibyendu Biswas.
         Version: 1.0
         Revisions: None
    """

    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Accuracy_Matrix_Logs.txt"  # this file path help to log the details in this file Executions_Logs/Training_Logs/Find_Best_Model_Logs.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def ForOnlyDT(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyDT
            Description: This method helps to get the model using Decision Tree after training.

            Output: model (decesion tree).
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the DecisionTree model")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.decision_tree = self.cls.CreateDecisionTreeClassifier(self.x_train, self.y_train)
            self.decision_tree.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.decision_tree_ypred_train = self.decision_tree.predict(self.x_train)
            # predection using test data:--
            self.decision_tree_ypred_test = self.decision_tree.predict(self.x_test)
            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(
                    self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using DecisionTree Classifier.
                self.decision_tree_score_test = accuracy_score(self.y_test, self.decision_tree_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using DecisionTree the Accuracy Score for test data of the model is: {str(self.decision_tree_score_test)}")
                # get the Accuracy score for train data using DecisionTree Classifier.
                self.decision_tree_score_train = accuracy_score(self.y_train, self.decision_tree_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using DecisionTree the Accuracy Score for train data of the model is: {str(self.decision_tree_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using DecisionTree Classifier.
                self.decision_tree_score_test = roc_auc_score(self.y_test, self.decision_tree_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using DecisionTree the AUC-ROC Score for test data of the model is: {str(self.decision_tree_score_test)}")
                # get the AUC-ROC score for train data using DecisionTree Classifier
                self.decision_tree_score_train = roc_auc_score(self.y_train, self.decision_tree_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using DecisionTree the AUC-ROC Score for train data of the model is: {str(self.decision_tree_score_train)}")
            # return the model and accuracy score.
            return self.decision_tree, self.decision_tree_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyRF(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyRF
            Description: This method helps to get the model using Random Forest after training.

            Output: model (random forest).
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the RandomForest model")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.random_forest = self.cls.CreateRandomForestClassifie(self.x_train, self.y_train)
            self.random_forest.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.random_forest_ypred_train = self.random_forest.predict(self.x_train)
            # predection using test data:--
            self.random_forest_ypred_test = self.random_forest.predict(self.x_test)
            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(
                    self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using RandomForest Classifier.
                self.random_forest_score_test = accuracy_score(self.y_test, self.random_forest_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using RandomForest the Accuracy Score for test data of the model is: {str(self.random_forest_score_test)}")
                # get the Accuracy score for train data using RandomForest Classifier.
                self.random_forest_score_train = accuracy_score(self.y_train, self.random_forest_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using RandomForest the Accuracy Score for train data of the model is: {str(self.random_forest_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using RandomForest Classifier.
                self.random_forest_score_test = roc_auc_score(self.y_test, self.random_forest_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using RandomForest the AUC-ROC Score for test data of the model is: {str(self.random_forest_score_test)}")
                # get the AUC-ROC score for train data using RandomForest Classifier
                self.random_forest_score_train = roc_auc_score(self.y_train, self.random_forest_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using RandomForest the AUC-ROC Score for train data of the model is: {str(self.random_forest_score_train)}")
            # return the model and accuracy score.
            return self.random_forest, self.random_forest_score_train

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyXGBoost(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyXGBoost
            Description: This method helps to get the model using XGBoost after training.

            Output: model (XGBoost).
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the XGBoost model")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.XGBoost = self.cls.CreateXGBoostClassifier(self.x_train, self.y_train)
            self.XGBoost.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.XGBoost_ypred_train = self.XGBoost.predict(self.x_train)
            # predection using test data:--
            self.XGBoost_ypred_test = self.XGBoost.predict(self.x_test)
            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(
                    self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using XGBoost Classifier.
                self.XGBoost_score_test = accuracy_score(self.y_test, self.XGBoost_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using XGBoost the Accuracy Score for test data of the model is: {str(self.XGBoost_score_test)}")
                # get the Accuracy score for train data using XGBoost Classifier.
                self.XGboost_score_train = accuracy_score(self.y_train, self.XGBoost_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using XGBoost the Accuracy Score for train data of the model is: {str(self.XGboost_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using XGBoost Classifier.
                self.XGBoost_score_test = roc_auc_score(self.y_test, self.XGBoost_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using XGBoost the AUC-ROC Score for test data of the model is: {str(self.XGBoost_score_test)}")
                # get the AUC-ROC score for train data using XGBoost Classifier
                self.XGBoost_score_train = roc_auc_score(self.y_train, self.XGBoost_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using XGBoost the AUC-ROC Score for train data of the model is: {str(self.XGBoost_score_train)}")
            # return the model and accuracy score.
            return self.XGBoost, self.XGBoost_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ForOnlyEnsembleDT(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyEnsembleDT
            Description: This method helps to get the model using Ensemble technique (base model is DecisionTree) after training.

            Output: model (Ensemble DecisionTree).
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the Decesion Tree model by applying Ensemble technique")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.Ensemble_decision_tree = self.cls.CreateEnsembleTechniquesDecisionTee(self.x_train, self.y_train)
            self.Ensemble_decision_tree.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.Ensemble_decision_tree_ypred_train = self.Ensemble_decision_tree.predict(self.x_train)
            # predection using test data:--
            self.Ensemble_decision_tree_ypred_test = self.Ensemble_decision_tree.predict(self.x_test)
            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(
                    self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_test = accuracy_score(self.y_test,
                                                                        self.Ensemble_decision_tree_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: DT) the Accuracy Score for test data of the model is: {str(self.Ensemble_decision_tree_score_test)}")
                # get the Accuracy score for train data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_train = accuracy_score(self.y_train,
                                                                         self.Ensemble_decision_tree_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: DT) the Accuracy Score for train data of the model is: {str(self.Ensemble_decision_tree_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_test = roc_auc_score(self.y_test,
                                                                       self.Ensemble_decision_tree_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: DT) the AUC-ROC Score of for test data the model is: {str(self.Ensemble_decision_tree_score_test)}")
                # get the AUC-ROC score for train data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_train = roc_auc_score(self.y_train,
                                                                        self.Ensemble_decision_tree_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: DT) the AUC-ROC Score for train data of the model is: {str(self.Ensemble_decision_tree_score_train)}")
            # return the model and accuracy score.
            return self.Ensemble_decision_tree, self.Ensemble_decision_tree_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyEnsembleKNN(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyEnsembleKNN
            Description: This method helps to get the model using Ensemble technique (base model is KNN) after training.

            Output: model (Ensemble KNN).
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the KNN model by applying Ensemble Technique")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.Ensemble_knn = self.cls.CreateEnsembleTechniquesKNN(self.x_train, self.y_train)
            self.Ensemble_knn.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.Ensemble_knn_ypred_train = self.Ensemble_knn.predict(self.x_train)
            # predection using test data:--
            self.Ensemble_knn_ypred_test = self.Ensemble_knn.predict(self.x_test)
            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(
                    self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_test = accuracy_score(self.y_test, self.Ensemble_knn_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: KNN) the Accuracy Score for test data of the model is: {str(self.Ensemble_knn_score_test)}")
                # get the Accuracy score for train data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_train = accuracy_score(self.y_train, self.Ensemble_knn_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: KNN) the Accuracy Score for train data of the model is: {str(self.Ensemble_knn_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_test = roc_auc_score(self.y_test, self.Ensemble_knn_ypred_test)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: KNN) the AUC-ROC Score for test data of the model is: {str(self.Ensemble_knn_score_test)}")
                # get the AUC-ROC score for train data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_train = roc_auc_score(self.y_train, self.Ensemble_knn_ypred_train)
                self.logger_object.log(self.file,
                                       f"Using Ensemble Technique (where base_model: KNN) the AUC-ROC Score for train data of the model is: {str(self.Ensemble_knn_score_train)}")
            # return the model and accuracy score
            return self.Ensemble_knn, self.Ensemble_knn_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ForClassificationALL(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForClassification
            Description: This method helps to get best model after training.

            Output: best model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the best model")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            # Decesion Tree
            self.dt, self.dt_score = self.ForOnlyDT(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # Random Forest
            self.rf, self.rf_score = self.ForOnlyRF(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # XgBoost
            self.xg, self.xg_score = self.ForOnlyXGBoost(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # Ensembel Technique, base model is Decision Tree
            self.endt, self.endt_score = self.ForOnlyEnsembleDT(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # Ensembel Technique, base model is KNN
            self.enknn, self.enknn_score = self.ForOnlyEnsembleKNN(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)

            # comparing between the XGBoost and RandomForest models based on accuracy and hen return the model:
            if self.xg_score > self.rf_score:
                return "XGBoost", self.xg, self.xg_score
            else:
                return "RandomForest", self.rf, self.rf_score

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyLinear(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyLinear
            Description: This method helps to get linear regression model after training.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the Linear model by applying Linear Regression")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.reg = Reg()  # call the regression model
            # create or load the model
            self.Linear_reg = self.reg.CreateLinearRegression(self.x_train, self.y_train)
            self.Linear_reg.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.Linear_reg_ypred_train = self.Linear_reg.predict(self.x_train)
            # predection using test data:--
            self.Linear_reg_ypred_test = self.Linear_reg.predict(self.x_test)
            # get the r2-score based on test data:
            self.Linear_r2_score_test = r2_score(self.y_test, self.Linear_reg_ypred_test)
            self.logger_object.log(self.file, f"Rsquare value for test data is {self.Linear_r2_score_test}")
            # get the r2-score based on train data:
            self.Linear_r2_score_train = r2_score(self.y_train, self.Linear_reg_ypred_train)
            self.logger_object.log(self.file, f"Rsquare value for train data is {self.Linear_r2_score_train}")
            # return the linear model and r2-score:
            return self.Linear_reg, self.Linear_r2_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ForOnlyRidge(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyRidge
            Description: This method helps to get ridge regression model after training.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the Linear model by applying Ridge Regression")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.reg = Reg()  # call the regression model
            # create or load the model
            self.Ridge_reg = self.reg.CreateRidgeRegression(self.x_train, self.y_train)
            self.Ridge_reg.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.Ridge_reg_ypred_train = self.Ridge_reg.predict(self.x_train)
            # predection using test data:--
            self.Ridge_reg_ypred_test = self.Ridge_reg.predict(self.x_test)
            # get the r2-score based on test data:
            self.Ridge_reg_score_test = r2_score(self.y_test, self.Ridge_reg_ypred_test)
            self.logger_object.log(self.file, f"Rsquare value for test data is {self.Ridge_reg_score_test}")
            # get the r2-score based on train data:
            self.Ridge_reg_score_train = r2_score(self.y_train, self.Ridge_reg_ypred_train)
            self.logger_object.log(self.file, f"Rsquare value for train data is {self.Ridge_reg_score_train}")
            # return the ridge model and r2-score:
            return self.Ridge_reg, self.Ridge_reg_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyLasso(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyLasso
            Description: This method helps to get lasso regression model after training.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the Linear model by applying Lasso Regression")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.reg = Reg()  # call the regression model
            # create or load the model
            self.Lasso_reg = self.reg.CreateLassoRegression(self.x_train, self.y_train)
            self.Lasso_reg.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.Lasso_reg_ypred_train = self.Lasso_reg.predict(self.x_train)
            # predection using test data:--
            self.Lasso_reg_ypred_test = self.Lasso_reg.predict(self.x_test)
            # get the r2-score based on test data:
            self.Lasso_reg_score_test = r2_score(self.y_test, self.Lasso_reg_ypred_test)
            self.logger_object.log(self.file, f"Rsquare value for test data is {self.Lasso_reg_score_test}")
            # get the r2-score based on train data:
            self.Lasso_reg_score_train = r2_score(self.y_train, self.Lasso_reg_ypred_train)
            self.logger_object.log(self.file, f"Rsquare value for train data is {self.Lasso_reg_score_train}")
            # return the lasso model and r2-score:
            return self.Lasso_reg, self.Lasso_reg_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ForOnlyElasticNet(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlyElasticNet
            Description: This method helps to get elastic regression model after training.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the Linear model by applying ElasticNet Regression")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.reg = Reg()  # call the regression model
            # create or load the model
            self.ElasticNet_reg = self.reg.CreateElasticNet(self.x_train, self.y_train)
            self.ElasticNet_reg.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.ElasticNet_reg_ypred_train = self.ElasticNet_reg.predict(self.x_train)
            # predection using test data:--
            self.ElasticNet_reg_ypred_test = self.ElasticNet_reg.predict(self.x_test)
            # get the r2-score based on test data:
            self.ElasticNet_reg_score_test = r2_score(self.y_test, self.ElasticNet_reg_ypred_test)
            self.logger_object.log(self.file, f"Rsquare value for test data is {self.ElasticNet_reg_score_test}")
            # get the r2-score based on train data:
            self.ElasticNet_reg_score_train = r2_score(self.y_train, self.ElasticNet_reg_ypred_train)
            self.logger_object.log(self.file, f"Rsquare value for train data is {self.ElasticNet_reg_score_train}")
            # return the lasso model and r2-score:
            return self.ElasticNet_reg, self.ElasticNet_reg_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ForOnlySVR(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForOnlySVR
            Description: This method helps to get SVR regression model after training.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the SVR model by applying SVR Regression")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            self.reg = Reg()  # call the regression model
            # create or load the model
            self.SVR = self.reg.CreateSVR(self.x_train, self.y_train)
            self.SVR.fit(self.x_train, self.y_train)
            # predection using train data:--
            self.SVR_ypred_train = self.SVR.predict(self.x_train)
            # predection using test data:--
            self.SVR_ypred_test = self.SVR.predict(self.x_test)
            # get the r2-score based on test data:
            self.SVR_score_test = r2_score(self.y_test, self.SVR_ypred_test)
            self.logger_object.log(self.file, f"Rsquare value for test data is {self.SVR_score_test}")
            # get the r2-score based on train data:
            self.SVR_score_train = r2_score(self.y_train, self.SVR_ypred_train)
            self.logger_object.log(self.file, f"Rsquare value for train data is {self.SVR_score_train}")
            # return the lasso model and r2-score:
            return self.SVR, self.SVR_score_test

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex



    def ForRegressionALL(self, x_train, x_test, y_train, y_test):
        """
            Method Name: ForRegressionALL
            Description: This method helps to get best regression model after training.

            Output: best model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, "It's help to get the best model")
            self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
            # Linear Regression:
            self.li, self.li_score = self.ForOnlyLinear(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # Ridge Regression:
            self.rid, self.rid_score = self.ForOnlyRidge(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # Lasso Regression:
            self.las, self.las_score = self.ForOnlyLasso(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # ElasticNet Regression:
            self.ela, self.ela_score = self.ForOnlyElasticNet(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)
            # SVR Regression:
            self.svr, self.svr_score = self.ForOnlySVR(x_train=self.x_train, x_test=self.x_test, y_train=self.y_train, y_test=self.y_test)

            # comparing the between the models using score:
            if self.ela_score > self.las_score:
                if self.ela_score > self.svr_score:
                    return "ElasticNet", self.ela, self.ela_score
                else:
                    return "SVR", self.svr, self.svr_score
            else:
                return "Lasso", self.las, self.las_score

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass

from Application_Log.logger import App_Logger
from Model_Creation.modelCreationRegression import Regression_Model_Finder as Reg
from Model_Creation.modelCreationClassification import To_Create_Classification_Model as Cls
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, confusion_matrix
import numpy as np
import pandas as pd


class Find_Best_Model:
    """
        This class shall be used for find the best model either classification or regression.

         Written By: Dibyendu Biswas.
         Version: 1.0
         Revisions: None
    """
    def __init__(self, file_path="Executions_Logs/Training_Logs/Find_Best_Model_Logs.txt"):
        self.file_path = file_path  # this file path help to log the details in this file Executions_Logs/Training_Logs/Find_Best_Model_Logs.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details


    def ForClassification(self, x_train, x_test, y_train, y_test):
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
            self.cls = Cls()  # call the classification models
            # create or load the model
            self.decision_tree = cls.CreateDecisionTreeClassifier(self.x_train, self.y_train)
            self.random_forest = cls.CreateRandomForestClassifie(self.x_train, self.y_train)
            self.XGBoost = cls.CreateXGBoostClassifier(self.x_train, self.y_train)
            self.Ensemble_decision_tree = cls.CreateEnsembleTechniquesDecisionTee(self.x_train, self.y_train)
            self.Ensemble_knn = cls.CreateEnsembleTechniquesKNN(self.x_train, self.y_train)
            # predection using train data:--
            self.decision_tree_ypred_train = self.decision_tree.predict(self.x_train)
            self.random_forest_ypred_train = self.random_forest.predict(self.x_train)
            self.XGBoost_ypred_train = self.XGBoost.predict(self.x_train)
            self.Ensemble_decision_tree_ypred_train = self.Ensemble_decision_tree.predict(self.x_train)
            self.Ensemble_knn_ypred_train = self.Ensemble_knn.predict(self.x_train)
            # predection using test data:--
            self.decision_tree_ypred_test = self.decision_tree.predict(self.x_test)
            self.random_forest_ypred_test = self.random_forest.predict(self.x_test)
            self.XGBoost_ypred_test = self.XGBoost.predict(self.x_test)
            self.Ensemble_decision_tree_ypred_test = self.Ensemble_decision_tree.predict(self.x_test)
            self.Ensemble_knn_ypred_test = self.Ensemble_knn.predict(self.y_test)

            # getting the Auc_Roc and Accuracy scores:
            if len(self.y_test.unique()) == 1 or len(self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using DecisionTree Classifier.
                self.decision_tree_score_test = accuracy_score(self.y_test, self.decision_tree_ypred_test)
                self.logger_object.log(self.file, f"Using DecisionTree the Accuracy Score of the model is: {str(self.decision_tree_score_test)}")
                # get the Accuracy score for train data using DecisionTree Classifier.
                self.decision_tree_score_train = accuracy_score(self.y_train, self.decision_tree_ypred_train)
                self.logger_object.log(self.file, f"Using DecisionTree the Accuracy Score of the model is: {str(self.decision_tree_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using DecisionTree Classifier.
                self.decision_tree_score_test = roc_auc_score(self.y_test, self.decision_tree_ypred_test)
                self.logger_object.log(self.file, f"Using DecisionTree the AUC-ROC Score of the model is: {str(self.decision_tree_score_test)}")
                # get the AUC-ROC score for train data using DecisionTree Classifier
                self.decision_tree_score_train = roc_auc_score(self.y_train, self.decision_tree_ypred_train)
                self.logger_object.log(self.file, f"Using DecisionTree the AUC-ROC Score of the model is: {str(self.decision_tree_score_train)}")


            if len(self.y_test.unique()) == 1 or len(self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using RandomForest Classifier.
                self.random_forest_score_test = accuracy_score(self.y_test, self.random_forest_ypred_test)
                self.logger_object.log(self.file, f"Using RandomForest the Accuracy Score of the model is: {str(self.random_forest_score_test)}")
                # get the Accuracy score for train data using RandomForest Classifier.
                self.random_forest_score_train = accuracy_score(self.y_train, self.random_forest_ypred_train)
                self.logger_object.log(self.file, f"Using RandomForest the Accuracy Score of the model is: {str(self.random_forest_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using RandomForest Classifier.
                self.random_forest_score_test = roc_auc_score(self.y_test, self.random_forest_ypred_test)
                self.logger_object.log(self.file, f"Using RandomForest the AUC-ROC Score of the model is: {str(self.random_forest_score_test)}")
                # get the AUC-ROC score for train data using RandomForest Classifier
                self.random_forest_score_train = roc_auc_score(self.y_train, self.random_forest_ypred_train)
                self.logger_object.log(self.file, f"Using RandomForest the AUC-ROC Score of the model is: {str(self.random_forest_score_train)}")


            if len(self.y_test.unique()) == 1 or len(self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using XGBoost Classifier.
                self.XGBoost_score_test = accuracy_score(self.y_test, self.XGBoost_ypred_test)
                self.logger_object.log(self.file, f"Using XGBoost the Accuracy Score of the model is: {str(self.XGBoost_score_test)}")
                # get the Accuracy score for train data using XGBoost Classifier.
                self.XGboost_score_train = accuracy_score(self.y_train, self.XGBoost_ypred_train)
                self.logger_object.log(self.file, f"Using XGBoost the Accuracy Score of the model is: {str(self.XGboost_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using XGBoost Classifier.
                self.XGBoost_score_test = roc_auc_score(self.y_test, self.XGBoost_ypred_test)
                self.logger_object.log(self.file, f"Using XGBoost the AUC-ROC Score of the model is: {str(self.XGBoost_score_test)}")
                # get the AUC-ROC score for train data using XGBoost Classifier
                self.XGBoost_score_train = roc_auc_score(self.y_train, self.XGBoost_ypred_train)
                self.logger_object.log(self.file, f"Using XGBoost the AUC-ROC Score of the model is: {str(self.XGBoost_score_train)}")


            if len(self.y_test.unique()) == 1 or len(self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_test = accuracy_score(self.y_test, self.Ensemble_decision_tree_ypred_test)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: DT) the Accuracy Score of the model is: {str(self.Ensemble_decision_tree_score_test)}")
                # get the Accuracy score for train data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_train = accuracy_score(self.y_train, self.Ensemble_decision_tree_ypred_train)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: DT) the Accuracy Score of the model is: {str(self.Ensemble_decision_tree_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_test = roc_auc_score(self.y_test, self.Ensemble_decision_tree_ypred_test)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: DT) the AUC-ROC Score of the model is: {str(self.Ensemble_decision_tree_score_test)}")
                # get the AUC-ROC score for train data using Ensemble Technique, where base model is DecisionTree Classifier.
                self.Ensemble_decision_tree_score_train = roc_auc_score(self.y_train, self.Ensemble_decision_tree_ypred_train)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: DT) the AUC-ROC Score of the model is: {str(self.Ensemble_decision_tree_score_train)}")


            if len(self.y_test.unique()) == 1 or len(self.y_train.unique()):  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                # get the Accuracy score for test data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_test = accuracy_score(self.y_test, self.Ensemble_knn_ypred_test)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: KNN) the Accuracy Score of the model is: {str(self.Ensemble_knn_score_test)}")
                # get the Accuracy score for train data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_train = accuracy_score(self.y_train, self.Ensemble_knn_ypred_train)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: KNN) the Accuracy Score of the model is: {str(self.Ensemble_knn_score_train)}")
            else:  # if there is more than one label then, we will get AUC-ROC score
                # get the AUC-ROC score for test data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_test = roc_auc_score(self.y_test, self.Ensemble_knn_ypred_test)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: KNN) the AUC-ROC Score of the model is: {str(self.Ensemble_knn_score_test)}")
                # get the AUC-ROC score for train data using Ensemble Technique, where base model is KNN.
                self.Ensemble_knn_score_train = roc_auc_score(self.y_train, self.Ensemble_knn_ypred_train)
                self.logger_object.log(self.file, f"Using Ensemble Technique (where base_model: KNN) the AUC-ROC Score of the model is: {str(self.Ensemble_knn_score_train)}")

            # get the best model by comparing the accuracy score for test data.
            # to create the list of score based on test data
            self.scores_test = [self.decision_tree_score_test, self.random_forest_score_test, self.XGBoost_score_test,
                                self.Ensemble_decision_tree_score_test, self.Ensemble_knn_score_test]
            # to create the list of score based on train data
            self.scores_train = [self.decision_tree_score_train, self.random_forest_score_train, self.XGboost_score_train,
                                 self.Ensemble_decision_tree_score_train, self.Ensemble_knn_score_train]
            # to create the list of models
            self.models = [self.decision_tree, self.random_forest, self.XGBoost, self.Ensemble_decision_tree, self.Ensemble_knn]
            # to create the index (model_names)
            self.index = ["Decision Tree", "Random Forest", "XGBoost", "Ensemble_DT", "Ensemble_KNN"]
            # to create the dictionary to store the test, train score with model
            self.score_data = {'Test_Score':self.scores_test, 'Train_Score':self.scores_train, 'model':self.models}
            # to create the pandas dataframe, and sorted by decending oder (because to get the best model)
            self.score_data = pd.DataFrame(self.score_data, index=self.index).sort_values(by='Test_Score', ascending=False)
            # get the best model for further prediction
            self.best_model = self.score_data['model'][0]
            # get the best model names
            self.best_model_name = self.score_data.index[0]
            return self.best_model_name, self.best_model  # return the best model name & best model



        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


from Application_Log.logger import App_Logger
from Model_Creation.modelCreationRegression import Regression_Model_Finder
from Model_Creation.modelCreationClassification import Classification_Model_Finder
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, confusion_matrix


class Find_Best_Model:
    """
        This class shall be used for find the best model either classification or regression.

         Written By: Dibyendu Biswas.
         Version: 1.0
         Revisions: None
    """
    def __init__(self, file_path):
        self.file_path = file_path  # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Model_Creation_Logs.txt"
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    

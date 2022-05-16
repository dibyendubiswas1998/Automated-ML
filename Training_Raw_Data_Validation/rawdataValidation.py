from Application_Log.logger import App_Logger
from scipy import stats
import numpy as np
import pandas as pd
import os
import re
import json
import shutil
from os import listdir

class Raw_Data_Validation:
    """
        This class shall be used for validate the raw data.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.logger_object = App_Logger()

    def CreateManualRegex(self):
        """
            Method Name: manualRegexCreation
            Description: This method contains a manually defined regex based on the given "FileName" .

            Output: Regex pattern
            On Failure: None

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def GetNeumericalFeatures(self, data):
        """
            Method Name: GetNeumericalFeatures
            Description: This method helps to get all the neumerical features.

            Output: neumerical features
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.neumericdata = data._get_numeric_data().columns
            if len(self.neumericdata) > 0:
                self.logger_object.log(file, f"Get all Neumeric data type {self.neumericdata}")
                file.close()
                return self.neumericdata
            else:
                self.logger_object.log(file, "Neumerical features are not found in dataset")
                file.close()
                return False

        except Exception as e:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {str(e)}")
            file.close()
            raise e


    def GetCatrgorycalFeatures(self, data):
        """
            Method Name: GetCatrgorycalFeatures
            Description: This method helps to get all the categorical features.

            Output: categorical features
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.categorical = data.dtypes[data.dtypes == 'object'].index
            if len(self.categorical) > 0:
                self.logger_object.log(file, f"Get all the Categorical data type: {self.categorical}")
                file.close()
                return self.categorical
            else:
                self.logger_object.log(file, "Categorical data are not present in dataset")
                file.close()
                return False

        except Exception as e:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {e}")
            file.close()
            raise e


    def GetLengthofData(self, data):
        """
            Method Name: GetLengthofData
            Description: This method helps to get the length (length of row & column) of the data.

            Output: length of data
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.row, self.col = data.shape[0], data.shape[1]
            if self.row > 0 or self.col > 0:
                self.logger_object.log(file, f"Get the length of data, rows:  {self.row}, columns: {self.col}")
                file.close()
                return self.row, self.col

            else:
                self.logger_object.log(file, "No data is present")
                file.close()
                return False

        except Exception as e:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {e}")
            file.close()
            raise e



    def IsMissingValuePresent(self, data):
        """
            Method Name: IsMissingValuePresent
            Description: This method helps to check is their any missing value present or not.

            Output: get the missing columns
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            missing_dataCol = []
            not_missing_dataCol = []
            for col in data.columns:
                if data[col].isnull().sum() > 0:
                    missing_dataCol.append(col)
                else:
                    not_missing_dataCol.append(col)

            self.logger_object.log(file, f"Missing value are present at {missing_dataCol}")
            self.logger_object.log(file, f"Missing values are not present at {not_missing_dataCol}")
            file.close()
            return missing_dataCol, not_missing_dataCol

        except Exception as e:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {e}")
            file.close()
            raise e


    def IsDataImbalanced(self, data, y):
        """
            Method Name: IsDataImbalanced
            Description: This method helps to check is data imbalanced or not.

            Output: True (if not balanced), False (if balanced)
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Check dataset is balanced or not: {dict(data[y].value_counts())}")
            vals = []
            for key, value in dict(data[y].value_counts()).items():
                vals.append(value)
            for i in range(len(vals)):
                if vals[i] == vals[i+1]:
                    self.logger_object.log(file, 'Dataset is balanced')
                    return True
                    break
                else:
                    self.logger_object.log(file, 'Dataset is not balanced')
                    return False
                    break
            file.close()

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex


    def IsOutliersPresent(self, data, cols, threshold):
        """
            Method Name: IsOutliersPresent
            Description: This method helps to check is outliers present in a particular column.
                         Here I use Z-Score method.

            Output: True (if present), False (if not present)
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            outliers_col = []
            for col in cols:
                z = np.abs(stats.zscore(data[col]))
                outliers_index = np.where(pd.DataFrame(z) > threshold)
                if len(outliers_index[0]) > 0:
                    outliers_col.append(col)
                    self.logger_object.log(file, f"Outliers are present at: {col} {outliers_index[0]}")
                else:
                    self.logger_object.log(file, f"Outliers are not present in dataset at: {col} {outliers_index[0]}")
            file.close()
            return outliers_col

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex



    def CreateDirectoryGoodBadData(self):
        """
            Method Name: CreateDirectoryGoodBadData
            Description: This method helps to create Good_Raw_Data & Bad_Raw_Data Data directory to store good and bad data
                         respectively .

            Output: create Good_Raw_Data, Bad_Raw_Data directory.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None.
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            path = os.path.join("../Training_Raw_Files_Validate/", "Good_Raw_Data/")
            if not os.path.isdir(path):
                os.makedirs(path)
                self.logger_object.log(file, "Good_Raw_Data directory is created")

            path = os.path.join("../Training_Raw_Files_Validate/", "Bad_Raw_Data/")
            if not os.path.isdir(path):
                os.makedirs(path)
                self.logger_object.log(file, "Bad_Raw_Data directory is created")
            file.close()

        except OSError as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex

    def DeleteExistingGoodRawDataTrainingFolder(self):
        """
            Method Name: DeleteExistingGoodRawDataTrainingFolder
            Description: This method deletes the directory made to store the Good Data
                         after loading the data in the table. Once the good files are
                         loaded in the DB,deleting the directory ensures space optimization.

            Output: None
            On Failure: OSError

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            path = "../Training_Raw_Files_Validate/"
            if os.path.isdir(path + "Good_Raw_Data/"):
                shutil.rmtree(path + "Good_Raw_Data/")
                file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
                self.logger_object.log(file, "Good_Raw_Data directory delete successfully")
                file.close()

        except OSError as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex


    def DeleteExistingBadRawDataTrainingFolder(self):
        """
            Method Name: DeleteExistingBadRawDataTrainingFolder
            Description: This method deletes the directory made to store the bad Data.

            Output: None
            On Failure: OSError

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            path = "../Training_Raw_Files_Validate/"
            if os.path.isdir(path + "Bad_Raw_Data/"):
                shutil.rmtree(path + "Bad_Raw_Data/")
                file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
                self.logger_object.log(file, "Bad_Raw_Data directory delete successfully")
                file.close()

        except OSError as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex


    def ValidateMissingValuesInWholeColumn(self):
        """
            Method Name: ValidateMissingValuesInWholeColumn
            Description: This function validates if any column in the csv file has all values missing.
                         If all the values are missing, the file is not suitable for processing.
                         SUch files are moved to bad raw data.

            Output: Drop column
            On Failure: Raise Error

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            for file in listdir("Training_Raw_Files_Validate/Good_Raw_Data/"):
                data = pd.read_csv("Training_Raw_Files_Validate/Good_Raw_Data/" + file)
                count = 0

                for column in data.columns:
                    if (len(data[column]) - data[column].count()) == len(data[column]):
                        count += 1
                        shutil.move("Training_Raw_Files_Validate/Good_Raw_Data/" + file,
                                    "Training_Raw_Files_Validate/Good_Raw_Data/")
                        self.logger.log(file, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % file)
                        break

                if count == 0:
                    data.to_csv("Training_Raw_Files_Validate/Good_Raw_Data/" + file, index=None, header=True)

        except OSError as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex

        except Exception as ex:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {ex}")
            file.close()
            raise ex


if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/boston.csv", 'csv', separator=',')
    print(data)

    validation = Raw_Data_Validation()
    validation.CreateDirectoryGoodBadData()
    # validation.DeleteExistingGoodRawDataTrainingFolder()
    # validation.DeleteExistingBadRawDataTrainingFolder()
    # print(result)



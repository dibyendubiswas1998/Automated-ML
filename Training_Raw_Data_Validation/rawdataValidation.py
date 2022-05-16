from Application_Log.logger import App_Logger
import numpy as np
import pandas as pd
import os
import re
import json
import shutil

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
            raise e


if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris1.csv", 'csv', separator=',')
    print(data)

    validation = Raw_Data_Validation()
    result = validation.IsDataImbalanced(data, 'species')
    print(result)



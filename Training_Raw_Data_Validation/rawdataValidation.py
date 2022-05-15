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
            self.logger_object.log(file, f"Get all Neumeric data type {self.neumericdata}")
            file.close()
            return self.neumericdata

        except Exception as e:
            file = open("../Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt", 'a+')
            self.logger_object.log(file, f"Error: {str(e)}")
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
            raise e




if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris.csv", 'csv', separator=None)
    print(data)

    validation = Raw_Data_Validation()
    categorycal = validation.GetCatrgorycalFeatures(data)
    print(categorycal)



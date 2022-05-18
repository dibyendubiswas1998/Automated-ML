from Application_Log.logger import App_Logger
from Training_Raw_Data_Validation.rawdataValidation import Raw_Data_Validation
from scipy import stats
import numpy as np
import pandas as pd
import os
import re
import json
import shutil
from os import listdir

class Data_Transformation:
    """
        This class shall be used for performing the data transformation operations.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_path):
        self.file_path = file_path  # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Data_Tansformation_Logs.txt"
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def ToReplaceMissingWithNull(self, data):
        """
            Method Name: replaceMissingWithNull
            Description: This method replaces the missing values in columns with "NULL" to
                          store in the table.

            Output: data (after replace with NULL)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            for self.column in self.data:
                self.data[self.column].fillna('NULL', inplace=True)  # replace the missing value with NULL, for store to database.
            self.logger_object.log(self.file, "Successfully replace with NULL value")
            self.file.close()
            return self.data  # return the data with NULL value (if missing values are present)

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToLogTransformation(self, data, Xcols=None):
        """
            Method Name: ToLogTransformation
            Description: This method helps to transform (using log) the data

            Output: data (after logarithmic transformation)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None

        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Xcols = Xcols
            if self.Xcols is None:  # check if column or columns are mention or not.
                for self.col in self.data:  # columns wise perform the operation
                    self.data[self.col] = np.log(self.data[self.col])  # to apply Log Transformation
                self.logger_object.log(self.file, f"Log Tranformation perform in columns {self.data.columns}")  # log the operation details
                self.file.close()
            else:
                for self.col in self.Xcols:  # columns wise perform the operation
                    self.data[self.col] = np.log(self.data[self.col])  # to apply Log Transformation
                self.logger_object.log(self.file, f"Log Tranformation perform in columns {self.Xcols}")  # log the operation details
                self.file.close()
            return self.data  # return data after compute Log Transformation.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToSquareRootTransformation(self, data, Xcols=None):
        """
            Method Name: ToSquareRootTransformation
            Description: This method helps to transform (using square root) the data

            Output: data (after square root transformation)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Xcols = Xcols
            if self.Xcols is None:  # check if column or columns are mention or not.
                for self.col in self.data:  # columns wise perform the operation
                    self.data[self.col] = np.sqrt(self.data[self.col])  # to apply the Square-Root Transformation.
                self.logger_object.log(self.file, f"Square Root Tranformation perform in columns {self.data.columns}")  # log the operations
                self.file.close()
            else:
                for self.col in self.Xcols:   # columns wise perform the operation
                    self.data[self.col] = np.sqrt(self.data[self.col])   # to apply the Square-Root Transformation.
                self.logger_object.log(self.file, f"Square Root Tranformation perform in columns {self.Xcols}")  # log the operations
                self.file.close()

            return self.data  # return data after compute Square-Root Transformation.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ToBoxCoXTransformation(self, data, Xcols=None):
        """
            Method Name: ToBoxCoXTransformation
            Description: This method helps to transform (using Box-Cox) the data

            Output: data (after Box-Cox transformation)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Xcols = Xcols
            if self.Xcols is None:   # check if column or columns are mention or not.
                for self.col in self.data:  # columns wise perform the operation
                    self.data[self.col], self.parameter = stats.boxcox(self.data[self.col])  # to apply the Box-Cox Transformation.
                self.logger_object.log(self.file, f"Box-Cox Tranformation perform in columns {self.data.columns}")  # log the details.
                self.file.close()

            else:
                for self.col in Xcols:  # columns wise perform the operation
                    self.data[self.col], parameter = stats.boxcox(self.data[self.col])   # to apply the Box-Cox Transformation.
                self.logger_object.log(self.file, f"Box-Cox Tranformation perform in columns {self.Xcols}")  # log the details.
                self.file.close()
            return self.data  # return data after compute Box-Cox Transformation.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex




if __name__ == '__main__':
    pass




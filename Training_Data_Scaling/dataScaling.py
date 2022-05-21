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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

class Data_Scaling:
    """
        This class shall be used for scalling the data set.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_path='Executions_Logs/Training_Logs/Data_Scaling_Logs.txt'):
        self.file_path = file_path   # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Data_Scaling_Logs.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details


    def ToNormalized(self, data):
        """
            Method Name: ToNormalized
            Description: This method helps to scale the data using MinMaxScaler() technique.

            Output: data (after normalized)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.minmax = MinMaxScaler()
            self.scaled_data = self.minmax.fit_transform(self.data)
            self.logger_object.log(self.file, f"Normalize the data using MinMaxScaler() technique")
            self.file.close()
            return self.scaled_data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToStandarized(self, data):
        """
            Method Name: ToStandarized
            Description: This method helps to scale the data using StandardScaler() technique.

            Output: data (after standarized)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.stadarize = StandardScaler()
            self.scaled_data = self.stadarize.fit_transform(self.data)
            self.logger_object.log(self.file, f"Standarized the data using StandardScaler() technique")
            self.file.close()
            return self.scaled_data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToQuantilTransformerScaler(self, data):
        """
            Method Name: ToStandarized
            Description: This method helps to scale the data using QuantilTransformerScaler() technique.

            Output: data (after scaling)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.quantile = QuantileTransformer()
            self.scaled_data = self.quantile.fit_transform(self.data)
            self.logger_object.log(self.file, f"Scaling the data using QuantileTransformer() technique")
            self.file.close()
            return self.scaled_data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass




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
    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/Data_Scaling_Logs.txt"
        self.logger_object = App_Logger()


    def ToNormalized(self, data, Xcols):
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
            file = open(self.file_path, 'a+')
            minmax = MinMaxScaler()
            scaled_data = minmax.fit_transform(data[Xcols])
            scaled_data = pd.DataFrame(scaled_data, columns=Xcols)
            self.logger_object.log(file, f"Normalize the data using MinMaxScaler() technique, columns: {Xcols}")
            file.close()
            return scaled_data, minmax

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex


    def ToStandarized(self, data, Xcols):
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
            file = open(self.file_path, 'a+')
            stadarize = StandardScaler()
            scaled_data = stadarize.fit_transform(data[Xcols])
            scaled_data = pd.DataFrame(scaled_data, columns=Xcols)
            self.logger_object.log(file, f"Normalize the data using StandardScaler() technique, columns: {Xcols}")
            file.close()
            return scaled_data, stadarize

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex


    def ToQuantilTransformerScaler(self, data, Xcols):
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
            file = open(self.file_path, 'a+')
            quantile = QuantileTransformer()
            scaled_data = quantile.fit_transform(data[Xcols])
            scaled_data = pd.DataFrame(scaled_data, columns=Xcols)
            self.logger_object.log(file, f"Normalize the data using QuantileTransformer() technique, columns: {Xcols}")
            file.close()
            return scaled_data, quantile

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex


if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/irisNull.csv", 'csv', separator=',')
    print(data.head(22))

    scaling = Data_Scaling()
    data, scale = scaling.ToQuantilTransformerScaler(data, data.columns)
    print(data.head(20))



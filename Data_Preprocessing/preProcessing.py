from Application_Log.logger import App_Logger
from scipy import stats
import numpy as np
import pandas as pd
import os
import re
import json
import shutil
from os import listdir
from sklearn.impute import KNNImputer


class Data_Preprocessing:
    """
        This class shall  be used to preprocess the data before training.

        Written By: Dibyendu Biswas
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/Data_Preprocessing_Logs.txt"
        self.logger_object = App_Logger()


    def ToDroColumns(self, data, Xcols=None):
        """
            Method Name: DroColumns
            Description: This method helps to drop the columns from dataset.

            Output: data (after droping columns or given data)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            file = open(self.file_path, 'a+')
            if Xcols is None:
                self.logger_object.log(file, "No columns are droped")
                file.close()
                return data
            else:
                data = data.drop(axis=1, columns=Xcols)
                self.logger_object.log(file, f"Successfully drop the columns: {Xcols}")
                file.close()
                return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex

    def ToSeparateTheLabelFeature(self, data, Ycol):
        """
            Method Name: ToSeparateTheLabelFeature
            Description: This method helps to separate the label column (X, Y)

            Output: feature_data(X), label_data(Y)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            file = open(self.file_path, 'a+')
            self.X = data.drop(axis=1, columns=Ycol)
            self.Y = data[Ycol]
            self.logger_object.log(file, f"Successfully drop the column {Ycol}")
            file.close()
            return self.X, self.Y

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex



if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris2.csv", 'csv', separator=',')
    print(data.head(22), '\n\n')

    preprocess = Data_Preprocessing()
    X, Y = preprocess.ToSeparateTheLabelFeature(data, 'species')
    print(X.head())
    print(Y)


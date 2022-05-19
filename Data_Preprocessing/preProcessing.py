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
    def __init__(self, file_path="Execution_Logs/Training_Logs/Data_Preprocessing.txt"):
        self.file_path = file_path   # this file path help to log the details in particular file = Execution_Logs/Training_Logs/Data_Preprocessing.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details


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
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Xcols = Xcols
            if self.Xcols is None:  # if you can't mention the column(s), then nothing happen
                self.logger_object.log(self.file, "No columns are droped")
                self.file.close()
                return self.data  # simply return the dta
            else:
                self.data = self.data.drop(axis=1, columns=self.Xcols)  # drop the column/ columns
                self.logger_object.log(self.file, f"Successfully drop the columns: {self.Xcols}")
                self.file.close()
                return self.data  # return data after drop column/ columns

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
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
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Ycol = Ycol
            self.X = self.data.drop(axis=1, columns=self.Ycol)  # separate the features columns
            self.Y = self.data[self.Ycol]  # separate the output or label column
            self.logger_object.log(self.file, f"Successfully drop the column {self.Ycol}")
            self.file.close()
            return self.X, self.Y   # return the features & output or label column(s)

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToImputeMissingValues(self, data):
        """
            Method Name: ToImputeMissingValues
            Description: This method replaces all the missing values in the Dataframe using
                         KNN Imputer.

            Output: data (after impute missing values)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)  # impute the missing value with KNNImputer
            self.new_data = self.imputer.fit_transform(self.data)
            self.new_data = pd.DataFrame(self.new_data, columns=self.data.columns)
            self.logger_object.log(self.file, "Impute the missing values with KNNImputer")
            self.file.close()
            return self.new_data  # return data where no missing values are present

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToGetColumnsWithZeroStandardDeviation(self, data):
        """
            Method Name: ToGetColumnsWithZeroStandardDeviation
            Description: This method finds out the columns which have a standard deviation of
                         zero.

            Output: columns (get the columns with zero standard deviation)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.data_describe = data.describe()
            self.droping_cols = []
            for self.col in self.data:
                if self.data_describe[self.col]['std'] == 0:  # to check the which column have standard deviation zero
                    self.droping_cols.append(self.col)   # append those columns where std is zero.
            if len(self.droping_cols) > 0:
                self.logger_object.log(self.file, f"Successfully get the Zero-Standard deviation columns {self.droping_cols}")
                self.file.close()
            else:
                self.logger_object.log(self.file, f"Not get the Zero-Standard deviation columns {self.droping_cols}")
                self.file.close()
            return self.droping_cols  # return the columns, if you want you can drop those columns.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex



if __name__ == '__main__':
    pass





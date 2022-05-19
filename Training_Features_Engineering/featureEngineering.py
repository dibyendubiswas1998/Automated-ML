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
from imblearn.over_sampling import BorderlineSMOTE, SMOTE

class Feature_Engineerings:
    """
         This class shall be used for handle the raw data.

         Written By: Dibyendu Biswas.
         Version: 1.0
         Revisions: None
    """
    def __init__(self, file_path="Executions_Logs/Training_Logs/Features_Engineering_Logs.txt"):
        self.file_path = file_path   # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Features_Engineering_Logs.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def ToHandleImbalancedData(self, data, ycol):
        """
            Method Name: ToHandleImbalancedData
            Description: This method helps to handle the imbalanced data.
                         Here, we use Borderline-SMOTE to handle the imbalance data.

            Output: data (after balance)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.ycol = ycol
            self.bsmote = BorderlineSMOTE(random_state=42, kind='borderline-1')  # use BorderLine SMOTE to oversample the data
            self.X = self.data.drop(axis=1, columns=[self.ycol])   # drop the output columns
            self.Y = data[self.ycol]
            self.x, self.y = self.bsmote.fit_resample(self.X, self.Y)
            self.data = self.x
            self.data[self.ycol] = pd.DataFrame(self.y, columns=[self.ycol])
            self.logger_object.log(self.file, "Handle the imbalanced data using Borderline-SMOTE")
            self.file.close()
            return self.data   # return data (with features & label/output) after oversampling

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToHandleOutliers(self, data, col, threshold=3):
        """
            Method Name: ToHandleOutliers
            Description: This method helps to handle the outliers using Z-Score.

            Output: data (after removing the outliers)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.col = col
            self.threshold = threshold  # bydefault we set the threshold value, i.e. 3
            self.z_score = np.abs(stats.zscore(self.data[self.col]))  # to apply the Z-Score to handle the outliers
            self.not_outliers_index = np.where(pd.DataFrame(self.z_score) < self.threshold)[0]  # get the indexes where outliers are not present or ignore outliers indexes
            self.data[self.col] = pd.DataFrame(self.data[self.col]).iloc[self.not_outliers_index]  # get the data without outliers
            self.logger_object.log(self.file, f"Successfully remove the outliers from data {self.col}")
            self.file.close()
            return self.data  # return the data without outliers.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToRemoveDuplicateValues(self, data):
        """
            Method Name: ToRemoveDuplicateValues
            Description: This method helps to remove the duplicate values

            Output: data (after removing the duplicate values)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.logger_object.log(self.file, f"Before drop the duplicates values, the shape of data is {self.data.shape}")
            self.data = self.data.drop_duplicates()  # simple drop the duplicates values from the given dataset
            self.logger_object.log(self.file, f"After drop the duplicates values, the shape of data is {self.data.shape}")
            self.logger_object.log(self.file, "Successfully drop the duplicates values")
            self.file.close()
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToMappingOutputCol(self, data, ycol):
        """
            Method Name: ToMappingOutputCol
            Description: This method helps to replace the categorical value to integer value.

            Output: data (after remplace integer value)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.ycol = ycol
            self.category = []
            if data[self.ycol].dtypes not in ['int', 'int64', 'int32', 'float', 'float32', 'float64']:   # if the label column is categorical data then simply to do map
                for self.cate in self.data[self.ycol].unique().tolist():
                    self.category.append(self.cate)
            self.logger_object.log(self.file, f"In output column has {self.category} categories.")
            self.value = list(range(len(self.data[self.ycol].unique())))
            self.dictionary = dict(zip(self.category, self.value))   # perform mapping like {'aa':0, 'bc':1, 'cd':3}.
            self.data[self.ycol] = self.data[self.ycol].map(self.dictionary)
            self.logger_object.log(self.file, f"Mapping operations is done successfully like this: {self.dictionary}")
            self.file.close()
            return self.data  # return data after mapping

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ToHandleMissingValues(self, data, Xcols=None):
        """
            Method Name: ToHandleMissingValues
            Description: This method helps to handle the missing values. Using this method we replace missing values
                         with mean (of that particular feature).

            Output: data (after handle missing values)
            On Failure: Raise Error.

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.Xcols = Xcols
            if self.Xcols is None:
                for self.col in self.data:  # check if column or columns are mention or not.
                    self.data[self.col].dropna(how='all', inplace=True)  # drop the row if all columns have missing value
                    self.data[self.col].fillna(self.data[self.col].mean(), inplace=True)   # replace the missing value with mean
                self.logger_object.log(self.file, f"Replace the missing value with mean value of {self.data.columns} columns")
                self.file.close()
            else:
                for self.col in self.Xcols:
                    self.data[self.col].dropna(how='all', inplace=True)  # drop the row if all columns have missing value
                    self.data[self.col].fillna(self.data[self.col].mean(), inplace=True)  # replace the missing value with mean
                self.logger_object.log(self.file, f"Replace the missing value with mean value of {self.Xcols} columns")
                self.file.close()
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass




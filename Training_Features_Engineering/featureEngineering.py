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
    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/Features_Engineering_Logs.txt"
        self.logger_object = App_Logger()

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
            file = open(self.file_path, 'a+')
            bsmote = BorderlineSMOTE(random_state=101, kind='borderline-1')
            X = data.drop(axis=1, columns=ycol)
            Y = data[ycol]
            x, y = bsmote.fit_resample(X, Y)
            data = x
            data[ycol] = pd.DataFrame(y, columns=[ycol])
            self.logger_object.log(file, "Handle the imbalanced data using Borderline-SMOTE")
            file.close()
            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
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
            file = open(self.file_path, 'a+')
            z_score = np.abs(stats.zscore(data[col]))
            not_outliers_index = np.where(pd.DataFrame(z_score) < threshold)[0]
            data[col] = pd.DataFrame(data[col]).iloc[not_outliers_index]
            self.logger_object.log(file, f"Successfully remove the outliers from data {col}")
            file.close()
            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
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
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Before drop the duplicates values, the shape of data is {data.shape}")
            data = data.drop_duplicates()
            self.logger_object.log(file, f"After drop the duplicates values, the shape of data is {data.shape}")
            self.logger_object.log(file, "Successfully drop the duplicates values")
            file.close()
            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
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
            file = open(self.file_path, 'a+')
            category = []
            if data[ycol].dtypes not in ['int', 'int64', 'int32', 'float', 'float32', 'float64']:
                for cate in data[ycol].unique().tolist():
                    category.append(cate)
            self.logger_object.log(file, f"In output column has {category} categories.")
            value = list(range(len(data[ycol].unique())))
            dictionary = dict(zip(category, value))
            data[ycol] = data[ycol].map(dictionary)
            self.logger_object.log(file, f"Mapping operations is done successfully like this: {dictionary}")
            file.close()
            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
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
            file = open(self.file_path, 'a+')

            if Xcols is None:
                for col in data:
                    data[col].dropna(how='all', inplace=True)
                    data[col].fillna(data[col].mean(), inplace=True)
                self.logger_object.log(file, f"Replace the missing value with mean value of {Xcols} columns")
                file.close()
            else:
                for col in Xcols:
                    data[col].dropna(how='all', inplace=True)
                    data[col].fillna(data[col].mean(), inplace=True)
                self.logger_object.log(file, f"Replace the missing value with mean value of {Xcols} columns")
                file.close()

            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex
        

if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/irisNull.csv", 'csv', separator=',')
    print(data.head(20))
    print(data.isnull().sum())

    featureEng = Feature_Engineerings()
    data = featureEng.ToHandleMissingValues(data)
    print(data.head(20))
    print(data.isnull().sum())



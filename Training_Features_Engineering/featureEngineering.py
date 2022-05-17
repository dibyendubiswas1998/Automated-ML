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

    

if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/boston.csv", 'csv', separator=',')
    print("Before drop the duplicate values:  ", data.shape)

    featureEng = Feature_Engineerings()
    data = featureEng.ToRemoveDuplicateValues(data)
    print("After drop the duplicate values:  ", data.shape)


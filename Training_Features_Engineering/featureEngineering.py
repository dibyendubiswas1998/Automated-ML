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


if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris2.csv", 'csv', separator=',')
    print(data['species'].value_counts())

    featureEng = Feature_Engineerings()
    data = featureEng.ToHandleImbalancedData(data, 'species')
    print(data['species'].value_counts())
    print(data.columns)

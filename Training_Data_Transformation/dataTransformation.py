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
    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/Data_Tansformation_Logs.txt"
        self.logger_object = App_Logger()

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
            file = open(self.file_path, 'a+')
            for column in data:
                data[column].fillna('NULL', inplace=True)
            self.logger_object.log(file, "Successfully replace with NULL value")
            file.close()
            return data

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex



if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris1.csv", 'csv', separator=',')
    print(data.head(22))

    dataTrans = Data_Transformation()
    data = dataTrans.ToReplaceMissingWithNull(data)
    print("after replacing with NULL \n\n\n", data.head(30))
    print(data.isnull().sum())


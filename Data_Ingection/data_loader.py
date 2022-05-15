import numpy as np
import pandas as pd
import string
from Application_Log.logger import App_Logger

class Data_Collection:
    """
        This class shall  be used for obtaining the data from the source for training.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self, file_object):
        self.file_object = file_object
        self.logger_object = App_Logger()

    def get_data(self, path, format, separator):
        """
            Method Name: get_data
            Description: This method reads the data from source this except only csv format.
            Output: A pandas DataFrame.
            On Failure: Raise Exception

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
         """
        try:
            self.logger_object.log(self.file_object, "Start to load the data")
            if format.lower() in 'csv':
                self.data = pd.read_csv(path, sep=separator)
                # print(self.data)

                self.logger_object.log(self.file_object, 'Data is Successfully load')
                return self.data
            else:
                self.logger_object.log(self.file_object, 'Data is not load Successfully')

        except Exception as e:
            self.logger_object.log(self.file_object, f'Data is not Successfully load: {e}')
            print(e)


if __name__ == '__main__':
    file = open("../Executions_Logs/Training_Logs/Genereal_Logs.txt", 'a+')
    path = "../Raw Data/winequality-red.csv"
    format = 'CSV'
    separator = '/t'
    datacol = Data_Collection(file)
    datacol.get_data(path, format, separator)

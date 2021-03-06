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
    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Genereal_Logs.txt"   # this file path help to log the details in particular file = Executions_Logs/Training_Logs/Genereal_Logs.txt
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def get_data(self, path, format=None, separator=None):
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
            self.file = open(self.file_path, 'a+')
            self.path = path
            self.format = format
            self.separator = separator
            if self.format is None:
                self.data = pd.read_csv(self.path, sep=self.separator)  # read the csv data
                self.logger_object.log(self.file, 'Data is Successfully load')
                self.file.close()

            if self.format.lower() in 'csv':  # check the data format is csv or not.
                self.data = pd.read_csv(self.path, sep=self.separator)  # read the csv data
                self.logger_object.log(self.file, 'Data is Successfully load')
                self.file.close()
                return self.data  # return the data


        except Exception as e:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f'Data is not Successfully load: {e}')
            self.file.close()
            raise e


if __name__ == '__main__':
    pass

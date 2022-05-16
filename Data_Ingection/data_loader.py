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
        self.file_path = "../Executions_Logs/Training_Logs/Genereal_Logs.txt"
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
            file = open(self.file_path, 'a+')
            if format.lower() in 'csv':
                self.data = pd.read_csv(path, sep=separator)
                self.logger_object.log(file, 'Data is Successfully load')
                return self.data
            else:
                self.logger_object.log(self.file_object, 'Data is not load Successfully')
            file.close()

        except Exception as e:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f'Data is not Successfully load: {e}')
            file.close()
            print(e)


if __name__ == '__main__':
    path = "../Raw Data/winequality-red.csv"
    format = 'CSV'
    separator = ';'
    datacol = Data_Collection()
    print(datacol.get_data(path, format, separator))

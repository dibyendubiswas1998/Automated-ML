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
        self.file_path = "../Executions_Logs/Training_Logs/Data_Scaling_Logs.txt"
        self.logger_object = App_Logger()


    


if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection
    data = Data_Collection().get_data("../Raw Data/iris2.csv", 'csv', separator=',')
    print(data.head(22))



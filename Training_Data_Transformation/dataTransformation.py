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
    


if __name__ == '__main__':
    pass


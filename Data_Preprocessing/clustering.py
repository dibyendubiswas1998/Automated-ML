import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Application_Log.logger import App_Logger
from kneed import KneeLocator

class KMeans_Clustering:
    """
        This class shall  be used to divide the data into clusters before training.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/Data_Preprocessing_Logs.txt"
        self.logger_object = App_Logger()




if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection

    data = Data_Collection().get_data("../Raw Data/boston.csv", 'csv', separator=',')
    print(data.head(15), '\n\n')



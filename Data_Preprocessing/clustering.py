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
        self.processing_data_path = "../Preprocessing_Data"
        self.logger_object = App_Logger()


    def Elbow_Method(self, data):
        """
            Method Name: Elbow_Method
            Description: This method saves the plot to decide the optimum number of clusters
                         to the file.

            Output: A picture saved to the directory
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            file = open(self.file_path, 'a+')
            wcss = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(data)
                wcss.append(kmeans.inertia_)

            plt.plot(range(1, 11), wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            plt.savefig(self.processing_data_path + '/' + "KMeans_Elbow.PNG")
            self.logger_object.log(file, f"Save the KMeans_Elbow graph in {self.processing_data_path} directory")
            self.kn = KneeLocator(range(1, 11), wcss, curve='convex', direction='decreasing')
            self.logger_object.log(file, f"Get the Number of clusters using KMeans Clustering, i.e. {self.kn.knee}")
            file.close()
            return self.kn.knee

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex
        



if __name__ == '__main__':
    from Data_Ingection.data_loader import Data_Collection

    data = Data_Collection().get_data("../Raw Data/boston.csv", 'csv', separator=',')
    print(data.head(15), '\n\n')

    clus = KMeans_Clustering()
    kn = clus.Elbow_Method(data)
    print(kn)


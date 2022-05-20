import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from Application_Log.logger import App_Logger
from kneed import KneeLocator
from File_Operations.fileMethods import File_Operations

class KMeans_Clustering:
    """
        This class shall  be used to divide the data into clusters before training.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """

    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Data_Preprocessing_Logs.txt"  # this file path help to log the details in particular file
        self.processing_data_path = "Preprocessing_Data"  # you can store to preprocess data before training path = Preprocessing_Data
        self.logger_object = App_Logger()  # call the App_Logger() to log the details
        self.fileOperation = File_Operations()

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
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.wcss = []
            for i in range(1, 11):
                self.kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                self.kmeans.fit(self.data)
                self.wcss.append(self.kmeans.inertia_)

            plt.plot(range(1, 11), self.wcss)  # creating the graph between WCSS and the number of clusters
            plt.title('The Elbow Method')
            plt.xlabel('Number of clusters')
            plt.ylabel('WCSS')
            # plt.show()
            plt.savefig(self.processing_data_path + '/' + "KMeans_Elbow.PNG")  # to save the graph (wcs vs no. of cluster)
            self.logger_object.log(self.file, f"Save the KMeans_Elbow graph in {self.processing_data_path} directory")
            self.kn = KneeLocator(range(1, 11), self.wcss, curve='convex', direction='decreasing')  # KneeLocator helps to get the number of cluster
            self.logger_object.log(self.file, f"Get the Number of clusters using KMeans Clustering, i.e. {self.kn.knee}")
            self.file.close()
            return self.kn.knee  # get or return the number of cluster

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ToCreateCluster(self, data, no_cluster):
        """
            Method Name: ToCreateCluster
            Description: This method helpa to create the clusters

            Output: datafram with cluster
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.no_cluster = no_cluster
            self.kmeans = KMeans(n_clusters=self.no_cluster, init='k-means++', random_state=101)  # create a cluster using KMeans Clustering
            self.fileOperation.ToSaveModel(model=self.kmeans, filename="KMeans Clustering")
            self.y_kmeans = self.kmeans.fit_predict(self.data)  # predict the cluster labels
            self.data['cluster_label'] = self.y_kmeans  # attach the cluster labels with the given data
            self.logger_object.log(self.file, "Successfully create the clusters & labeled the cluster")
            self.file.close()
            return self.data  # return the cluster labeled data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass


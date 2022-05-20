import pickle
import os
import shutil
from Application_Log.logger import App_Logger


class File_Operations:
    """
        This class shall be used to save the model after training
        and load the saved model and find correct model for prediction.

        Written By: Dibyendu Biswas
        Version: 1.0
        Revisions: None
    """

    def __init__(self, file_path="Executions_Logs/Training_Logs/File_Operations_Logs.txt"):
        self.file_path = file_path   # this file path help to log the details in particular file "Executions_Logs/Training_Logs/File_Operations_Logs.txt"
        self.logger_object = App_Logger()   # call the App_Logger() to log the details
        self.model_directory = "Models/"   # this directoty helps to store model in folder where path = Models/

    def ToSaveModel(self, model, filename):
        """
            Method Name: ToSaveModel
            Description: This method helpa to save the model file to directory

            Output: file get saved
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.model = model
            self.filename = filename
            self.path = os.path.join(self.model_directory, self.filename)
            if os.path.isdir(self.path):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)  # remove existing directory
                os.makedirs(self.path)   # create directory
            else:
                os.makedirs(self.path)
            with open(self.path + '/' + self.filename + '.sav', 'wb') as f:
                pickle.dump(self.model, f)  # to dump or save the model in a directory
            self.logger_object.log(self.file, f"{self.filename} Successfully drop the model to {self.model_directory}")
            self.file.close()
            return "success"

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def ToLoadModel(self, filename):
        """
            Method Name: ToLoadModel
            Description: This method helpa to load the model to memory.

            Output: model.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.filename = filename
            with open(self.model_directory + '/' + self.filename + '.sav', 'rb') as f:   # to load the particular model
                self.logger_object.log(self.file, "Model is successfully loaded")
                self.file.close()
                return pickle.load(f)  # return the particular model

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.logger_object.log(self.file, "Model is not load")
            self.file.close()
            raise ex

    def ToFindCorrectModel(self, cluster_number):
        """
            Method Name: ToFindCorrectModel
            Description: This method helpa to find the correct model.

            Output: model file.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)

            for self.fl in self.list_of_files:
                try:
                    if self.fl.index(str(self.cluster_number)) != -1:  # find the current model
                        self.model_name = self.fl
                except:
                    continue

            self.model_name = self.model_name.split('.')[0]
            self.logger_object.log(self.file, "Successfully find the correct Model")
            self.file.close()
            return self.model_name  # return model name for loading

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ToDownloadData(self):
        """

        :return:
        """
        pass

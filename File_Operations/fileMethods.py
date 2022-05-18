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

    def __init__(self):
        self.file_path = "../Executions_Logs/Training_Logs/File_Operations.txt"
        self.logger_object = App_Logger()
        self.model_directory = "../Models/"

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
            file = open(self.file_path, 'a+')
            path = os.path.join(self.model_directory, filename)
            if os.path.isdir(path):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)
            with open(path + '/' + filename + '.sav', 'wb') as f:
                pickle.dump(model, f)
            self.logger_object.log(file, f"{filename} Successfully drop the model to {self.model_directory}")
            file.close()
            return "success"

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
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
            file = open(self.file_path, 'a+')
            with open(self.model_directory + filename + '/' + filename + '.sav', 'rb') as f:
                self.logger_object.log(file, "Model is successfully loaded")
                return pickle.load(f)

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            self.logger_object.log(file, "Model is not load")
            file.close()
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
            file = open(self.file_path, 'a+')
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)

            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue

            self.model_name = self.model_name.split('.')[0]
            self.logger_object.log(file, "Successfully find the correct Model")
            file.close()
            return self.model_name

        except Exception as ex:
            file = open(self.file_path, 'a+')
            self.logger_object.log(file, f"Error is: {ex}")
            file.close()
            raise ex

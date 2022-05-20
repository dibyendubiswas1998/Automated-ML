from Application_Log.logger import App_Logger
from scipy import stats
import numpy as np
import pandas as pd
import os
import re
import json
import shutil
from os import listdir

class Raw_Data_Validation:
    """
        This class shall be used for validate the raw data.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt"   # this file path help to log the details in particular file =Executions_Logs/Training_Logs/Raw_Data_Validation_Logs.txt"
        self.logger_object = App_Logger()  # call the App_Logger() to log the details

    def CreateManualRegex(self):
        """
            Method Name: manualRegexCreation
            Description: This method contains a manually defined regex based on the given "FileName" .

            Output: Regex pattern
            On Failure: None

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        regex = "['wafer']+['\_'']+[\d_]+[\d]+\.csv"
        return regex

    def GetNeumericalFeatures(self, data):
        """
            Method Name: GetNeumericalFeatures
            Description: This method helps to get all the neumerical features.

            Output: neumerical features
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.neumericdata = self.data._get_numeric_data().columns   # get the neumeric columns from given dataset.
            if len(self.neumericdata) > 0:
                self.logger_object.log(self.file, f"Get all Neumeric data type {self.neumericdata}")
                self.file.close()
                return self.neumericdata  # if present, then return those neumeric columns
            else:
                self.logger_object.log(self.file, "Neumerical features are not found in dataset")
                self.file.close()
                return False   # if nor present then return False

        except Exception as e:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {str(e)}")
            self.file.close()
            raise e


    def GetCatrgorycalFeatures(self, data):
        """
            Method Name: GetCatrgorycalFeatures
            Description: This method helps to get all the categorical features.

            Output: categorical features
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.categorical = self.data.dtypes[self.data.dtypes == 'object'].index  # return categorical columns from give dataset.
            if len(self.categorical) > 0:
                self.logger_object.log(self.file, f"Get all the Categorical data type: {self.categorical}")
                self.file.close()
                return self.categorical   # if present, then return those categorical columns
            else:
                self.logger_object.log(self.file, "Categorical data are not present in dataset")
                self.file.close()
                return False  # if not, then return False

        except Exception as e:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {e}")
            self.file.close()
            raise e


    def GetLengthofData(self, data):
        """
            Method Name: GetLengthofData
            Description: This method helps to get the length (length of row & column) of the data.

            Output: length of data
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.row_length, self.col_length = self.data.shape[0], self.data.shape[1]  # get the row length & column length
            if self.row_length > 0 or self.col_length > 0:
                self.logger_object.log(self.file, f"Get the length of data, rows:  {self.row_length}, columns: {self.col_length}")
                self.file.close()
                return self.row_length, self.col_length  # return lenghth of row & col

            else:
                self.logger_object.log(self.file, "No data is present")
                self.file.close()
                return False, False

        except Exception as e:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {e}")
            self.file.close()
            raise e



    def IsMissingValuePresent(self, data):
        """
            Method Name: IsMissingValuePresent
            Description: This method helps to check is their any missing value present or not.

            Output: get the missing columns
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.missing_dataCol = []  # here add only those columns where missing values are present
            self.not_missing_dataCol = []  # here add only those columns where missing values are not present
            for self.col in self.data.columns:
                if self.data[self.col].isnull().sum() > 0:  # check (columns wise, one-by-one) if missing value present
                    self.missing_dataCol.append(self.col)  # append those columns where missing values present.
                else:
                    self.not_missing_dataCol.append(self.col)  # append those columns where missing values are not present

            if len(self.missing_dataCol) > 0:
                self.logger_object.log(self.file, f"Missing value are present at {self.missing_dataCol}")
                self.file.close()
                return self.missing_dataCol  # return only missing columns.
            else:
                self.logger_object.log(self.file, "Missing value are not present in dataset")
                self.file.close()
                return self.not_missing_dataCol  # return those columns where missing values are not present


        except Exception as e:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {e}")
            self.file.close()
            raise e


    def IsDataImbalanced(self, data, y):
        """
            Method Name: IsDataImbalanced
            Description: This method helps to check is data imbalanced or not.

            Output: True (if not balanced), False (if balanced)
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.y = y
            self.logger_object.log(self.file, f"Check dataset is balanced or not: {dict(self.data[self.y].value_counts())}")
            self.vals = []
            for self.key, self.value in dict(self.data[self.y].value_counts()).items():
                self.vals.append(self.value)
            for self.i in range(len(self.vals)):
                if self.vals[self.i] == self.vals[self.i+1]:  # check the data is balance or not.
                    self.logger_object.log(self.file, 'Dataset is balanced')
                    return True  # if balance then return True
                    break
                else:
                    self.logger_object.log(self.file, 'Dataset is not balanced')
                    return False  # if not balance then return False
                    break
            self.file.close()

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex


    def IsOutliersPresent(self, data, cols, threshold=3):
        """
            Method Name: IsOutliersPresent
            Description: This method helps to check is outliers present in a particular column.
                         Here I use Z-Score method.

            Output: True (if present), False (if not present)
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None

        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.cols = cols
            self.threshold = threshold
            self.outliers_col = []  # this is used to append the outliers columns.
            for self.col in self.cols:
                self.z = np.abs(stats.zscore(self.data[self.col]))  # to apply the Z-Score for getting the outliers one-by-one columns.
                self.outliers_index = np.where(pd.DataFrame(self.z) > self.threshold)  # get the outliers indexs.
                if len(self.outliers_index[0]) > 0:
                    self.outliers_col.append(self.col)  # appen the columns where outliers are present.
                    self.logger_object.log(self.file, f"Outliers are present at: {self.col} {self.outliers_index[0]}")
                else:
                    self.logger_object.log(self.file, f"Outliers are not present in dataset at: {self.col} {self.outliers_index[0]}")
            self.file.close()
            return self.outliers_col  # return only those columns where outliers are present.

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex



    def CreateDirectoryGoodBadData(self, directory):
        """
            Method Name: CreateDirectoryGoodBadData
            Description: This method helps to create Good_Raw_Data & Bad_Raw_Data Data directory to store good and bad data
                         respectively .

            Output: create Good_Raw_Data, Bad_Raw_Data directory.
            On Failure: Raise Error.

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None.
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.directory = directory
            self.path = os.path.join(self.directory + '/', "Good_Raw_Data/")  # mention the path to create the directory for Good_Raw_Data.
            if not os.path.isdir(self.path):
                os.makedirs(self.path)  # if Good_Raw_Data directory is not present then created
                self.logger_object.log(self.file, "Good_Raw_Data directory is created")

            self.path = os.path.join(self.directory + '/', "Bad_Raw_Data/")  # mention the path to create the directory for Bad_Raw_Data.
            if not os.path.isdir(self.path):
                os.makedirs(self.path)   # if Bad_Raw_Data directory is not present then created
                self.logger_object.log(self.file, "Bad_Raw_Data directory is created")
            self.file.close()

        except OSError as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex

    def DeleteExistingGoodRawDataTrainingFolder(self, directory):
        """
            Method Name: DeleteExistingGoodRawDataTrainingFolder
            Description: This method deletes the directory made to store the Good Data
                         after loading the data in the table. Once the good files are
                         loaded in the DB,deleting the directory ensures space optimization.

            Output: None
            On Failure: OSError

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.directory = directory
            self.path = self.directory + "/"
            if os.path.isdir(self.path + "Good_Raw_Data/"):
                shutil.rmtree(self.path + "Good_Raw_Data/")  # delete the Good_Raw_Data directory, if this directory is present.
                self.file = open(self.file_path, 'a+')
                self.logger_object.log(self.file, "Good_Raw_Data directory delete successfully")
                self.file.close()

        except OSError as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex


    def DeleteExistingBadRawDataTrainingFolder(self, directory):
        """
            Method Name: DeleteExistingBadRawDataTrainingFolder
            Description: This method deletes the directory made to store the bad Data.

            Output: None
            On Failure: OSError

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.directory = directory
            self.path = self.directory + "/"
            if os.path.isdir(self.path + "Bad_Raw_Data/"):
                shutil.rmtree(self.path + "Bad_Raw_Data/")  # delete the Bad_Raw_Data directory, if this directory is present.
                self.file = open(self.file_path, 'a+')
                self.logger_object.log(self.file, "Bad_Raw_Data directory delete successfully")
                self.file.close()

        except OSError as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex


    def ValidateMissingValuesInWholeColumn(self):
        """
            Method Name: ValidateMissingValuesInWholeColumn
            Description: This function validates if any column in the csv file has all values missing.
                         If all the values are missing, the file is not suitable for processing.
                         SUch files are moved to bad raw data.

            Output: Drop column
            On Failure: Raise Error

            Written By: Dibyendu Biswas.
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            for self.fl in listdir("Training_Raw_Files_Validate/Good_Raw_Data/"):
                self.data = pd.read_csv("Training_Raw_Files_Validate/Good_Raw_Data/" + self.fl)
                self.count = 0

                for self.column in self.data.columns:
                    if (len(self.data[self.column]) - self.data[self.column].count()) == len(self.data[self.column]):
                        self.count += 1
                        shutil.move("Training_Raw_Files_Validate/Good_Raw_Data/" + self.fl,
                                    "Training_Raw_Files_Validate/Good_Raw_Data/")
                        self.logger.log(self.file, "Invalid Column Length for the file!! File moved to Bad Raw Folder :: %s" % self.file)
                        break

                if self.count == 0:
                    self.data.to_csv("Training_Raw_Files_Validate/Good_Raw_Data/" + self.fl, index=None, header=True)

        except OSError as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    pass




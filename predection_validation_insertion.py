from datetime import datetime
from Application_Log.logger import App_Logger
from Predection_Raw_Data_Validation.predectionRawDataValidation import Predection_Raw_Data_Validation as Raw_Data_Validation
from Predection_Features_Engineering.predectionFeaturesEngineering import Predection_Feature_Engineerings as Feature_Engineerings
from Predection_Data_Transformation.predectionDataTransformation import Predection_Data_Transformation as Data_Transformation
from Predection_Data_Scaling.predectionDataScaling import Predection_Data_Scaling as Data_Scaling
from Data_Ingection.data_loader import Data_Collection
from Data_Preprocessing.preProcessing import Data_Preprocessing
import numpy as np
import pandas as pd


class Predection_Validation_Insertion:
    """
        This class shall be used for validate the predection data before predection.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.file_path = "Executions_Logs/Predection_logs/Predection_Main_Logs.txt"
        self.logger_object = App_Logger()
        self.raw_data = Raw_Data_Validation()
        self.fea_eng = Feature_Engineerings()
        self.data_trans = Data_Transformation()
        self.data_scl = Data_Scaling()
        self.pre_processing = Data_Preprocessing()


    def ValidateDataForPredection(self, data, yCol, outlier_threshold=3, imputeMissing='KNNImputer', dataTransformationType=None):
        """
            Method Name: ValidatePredectionData_Classification
            Description: This method helps to validate the training data for classification before start predection

            Output: good data
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.ycol = yCol
            self.xcol = data.drop(axis=1, columns=[self.ycol]).columns
            self.outlier_threshold = outlier_threshold
            self.imputeMissing = imputeMissing
            self.dataTransformationType = dataTransformationType
            # get the neumeric columns.
            self.neumeric_cols = self.raw_data.GetNeumericalFeatures(data=self.data)
            if len(self.neumeric_cols) > 0:
                self.logger_object.log(self.file, f"Neumeric columns are:  f{str(self.neumeric_cols)}")
            # get the categorical columns.
            self.categorical_cols = self.raw_data.GetCatrgorycalFeatures(data=self.data)
            if len(self.categorical_cols) > 0:
                self.logger_object.log(self.file, f"Categorical columns are:   f{str(self.categorical_cols)}")
            # get the length of data (row & columns)
            self.row_len, self.col_len = self.raw_data.GetLengthofData(data=self.data)
            self.logger_object.log(self.file, f"Length of data is {str(self.row_len)} {str(self.col_len)}")
            # get the missing columns
            self.missing_data_col = self.raw_data.IsMissingValuePresent(data=self.data)
            # get the outliers columns
            self.isoutlier_cols = self.raw_data.IsOutliersPresent(self.data, self.xcol,
                                                                  threshold=self.outlier_threshold)
            # is data balance (True or False)
            self.isbalance = self.raw_data.IsDataImbalanced(data=self.data, y=self.ycol)

            # Start the validation:
            #  Handle the missing values based on condition:
            if len(self.missing_data_col) > 0:
                self.logger_object.log(self.file, f"Missing values are present at {self.missing_data_col}")
                if self.imputeMissing.lower() == 'knnimputer':  # impute missing values with KNNImputer
                    self.data = self.pre_processing.ToImputeMissingValues(data=self.data)
                    self.logger_object.log(self.file, "Successfully handle the missing values by KNNImputer")
                else:  # impute missing values with mean value
                    self.data = self.fea_eng.ToHandleMissingValues(data=self.data, Xcols=self.xcol)
                    self.logger_object.log(self.file,
                                           "Successfully handle the missing values by mean value of that column respectively")
            else:
                self.logger_object.log(self.file, "Missing values are not present")
            #  Remove the duplicated values:
            self.data = self.fea_eng.ToRemoveDuplicateValues(data=self.data)
            self.logger_object.log(self.file, "Successfully remove the duplicate values")
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, after remove the duplicate values")
            #  Remove the Outliers:
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, before remove the outliers")
            if len(self.isoutlier_cols) > 0:
                self.logger_object.log(self.file, f"Get the outliers columns:  {self.isoutlier_cols}")
                for self.col in self.isoutlier_cols:  # one by one column remove the outliers
                    self.data = self.fea_eng.ToHandleOutliers(data=self.data, col=self.col,
                                                              threshold=self.outlier_threshold)
                self.logger_object.log(self.file, "Successfully remove the outliers")
            else:
                self.logger_object.log(self.file, "Their is no outliers in the data set")
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, after remove the outliers")
            # No need to balance the data for predection
            #  Again Handle the missing values based on condition:
            if len(self.missing_data_col) > 0:
                self.logger_object.log(self.file, f"Missing values are present at {self.missing_data_col}")
                if self.imputeMissing.lower() == 'knnimputer':  # impute missing values with KNNImputer
                    self.data = self.pre_processing.ToImputeMissingValues(data=self.data)
                    self.logger_object.log(self.file, "Successfully handle the missing values by KNNImputer")
                else:  # impute missing values with mean value
                    self.data = self.fea_eng.ToHandleMissingValues(data=self.data, Xcols=self.xcol)
                    self.logger_object.log(self.file,
                                           "Successfully handle the missing values by mean value of that column respectively")
            else:
                self.logger_object.log(self.file, "Missing values are not present")
            """  Perform Data Transformation Steps based on conditions  """
            if self.dataTransformationType is None:
                # return the good and clean data for classification problem
                return self.data
            # Log Transformation:
            if self.dataTransformationType.lower() in ['log', 'log transformation', 'logtransformation', 'logtrans']:
                self.data = self.data_trans.ToLogTransformation(data=self.data, Xcols=self.xcol)
                # return the good and clean data for classification problem
                return self.data
            # Square Root Transformation:
            if self.dataTransformationType.lower() in ['sqrt', 'square root transformation', 'squareroottransformation', 'square root']:
                self.data = self.data_trans.ToSquareRootTransformation(data=self.data, Xcols=self.xcol)
                # return the good and clean data for classification problem
                return self.data
            # Box-Cox Transformation:
            if self.dataTransformationType.lower() in ['boxcox', 'box cox transformation', 'boxcoxtransformation', 'box cox']:
                self.data = self.data_trans.ToBoxCoXTransformation(data=self.data, Xcols=self.xcol)
                # return the good and clean data for classification problem
                return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex




if __name__ == '__main__':
    pass



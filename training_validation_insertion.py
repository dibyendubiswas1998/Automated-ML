from datetime import datetime
from Application_Log.logger import App_Logger
from Training_Raw_Data_Validation.rawdataValidation import Raw_Data_Validation
from Training_Features_Engineering.featureEngineering import Feature_Engineerings
from Training_Data_Transformation.dataTransformation import Data_Transformation
from Training_Data_Scaling.dataScaling import Data_Scaling
from Data_Ingection.data_loader import Data_Collection
from Data_Preprocessing.preProcessing import Data_Preprocessing

class Training_Validation_Insertion:
    """
        This class shall be used for validate the training data before training.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """
    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Training_Main_Log.txt"
        self.logger_object = App_Logger()
        self.raw_data = Raw_Data_Validation()
        self.fea_eng = Feature_Engineerings()
        self.data_trans = Data_Transformation()
        self.data_scl = Data_Scaling()
        self.pre_processing = Data_Preprocessing()


    def ValidateTrainingData_Classification(self, data, yCol, outlier_threshold=3, imputeMissing='KNNImputer'):
        """
            Method Name: ValidateTrainingData_Classification
            Description: This method helps to validate the training data for classification before start training

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
            #  Mapping the output column:
            if self.data[self.ycol].dtypes not in ['int', 'int64', 'int32', 'float', 'float32', 'float64']:
                self.data = self.fea_eng.ToMappingOutputCol(data=self.data, ycol=self.ycol)  # use KNN imputer to impute missing values
                self.logger_object.log(self.file, "Successfully mapping the output columns using KNN imputer")
            else:
                self.logger_object.log(self.file, "No need to mapped the output columns")
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
            #  Balance the data:
            if not self.isbalance:
                self.logger_object.log(self.file, "Data set is not balanced")
                self.data = self.fea_eng.ToHandleImbalancedData(data=self.data, ycol=self.ycol)
                self.logger_object.log(self.file, "Successfully balanced the data set")
            else:
                self.logger_object.log(self.file, "Dataset is balanced, no need to balance again")
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, after balanced the dataset")
            # return the good and clean data for classification problem
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


    def ValidateTrainingData_Regression(self, data, yCol, outlier_threshold=3, imputeMissing='KNNImputer'):
        """
            Method Name: ValidateTrainingData_Regression
            Description: This method helps to validate the training data for regression before start training

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
            self.isoutlier_cols = self.raw_data.IsOutliersPresent(self.data, self.xcol, threshold=self.outlier_threshold)

            # Start the validation:
            #  Handle the missing values based on condition:
            if len(self.missing_data_col) > 0:
                self.logger_object.log(self.file, f"Missing values are present at {self.missing_data_col}")
                if self.imputeMissing.lower() == 'knnimputer':   # impute missing values with KNNImputer
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
            self.logger_object.log(self.file,
                                   f"Shape of dataset is {str(self.data.shape)}, after remove the duplicate values")
            #  Remove the Outliers:
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, before remove the outliers")
            if len(self.isoutlier_cols) > 0:
                self.logger_object.log(self.file, f"Get the outliers columns:  {self.isoutlier_cols}")
                for self.col in self.isoutlier_cols:  # one by one column remove the outliers
                    self.data = self.fea_eng.ToHandleOutliers(data=self.data, col=self.col, threshold=self.outlier_threshold)
                self.logger_object.log(self.file, "Successfully remove the outliers")
            else:
                self.logger_object.log(self.file, "Their is no outliers in the data set")
            self.logger_object.log(self.file, f"Shape of dataset is {str(self.data.shape)}, after remove the outliers")
            # return the good and clean data for regression problem
            return self.data

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex



if __name__ == '__main__':
    pass

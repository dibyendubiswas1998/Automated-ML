from Application_Log.logger import App_Logger
from File_Operations.fileMethods import File_Operations
from Data_Ingection.data_loader import Data_Collection
from Data_Preprocessing.preProcessing import Data_Preprocessing
from Training_Data_Scaling.dataScaling import Data_Scaling
from Data_Preprocessing.clustering import KMeans_Clustering
from training_validation_insertion import Training_Validation_Insertion
from predection_validation_insertion import Predection_Validation_Insertion
from Finding_Best_Model.findBestModel import Find_Best_Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Automated_ML:
    """
        This class shall be used for train the model.

        Written By: Dibyendu Biswas.
        Version: 1.0
        Revisions: None
    """

    def __init__(self):
        self.file_path = "Executions_Logs/Training_Logs/Model_Training_Log.txt"
        self.logger_object = App_Logger()
        self.data_load = Data_Collection()
        self.pre_processing = Data_Preprocessing()
        self.scaling = Data_Scaling()
        self.clustering = KMeans_Clustering()
        self.validation = Training_Validation_Insertion()
        self.validationP = Predection_Validation_Insertion()
        self.model1 = Find_Best_Model()

    def training(self, data_path, yCol, format='csv', separator=',', problemType='classification',
                 imputeMissing='KNNImputer',
                 outlier_threshold=3, handle_outliers=True, mapping_ycol=False, balance_data=False,
                 dropCols=None, scalingType=None, dataTransformationType=None, chooseAlgorithm='dt'):
        """
            Method Name: training
            Description: This method helps to train the model

            Output: best model
            On Failure: Raise Error


            :param    data_path: 'E/User1/my_folder/data.csv' (mention only path where data is present),\n
            :param    yCol: 'output_col' (mention output column or output feature),\n
            :param     format: 'csv' (mention data format like csv, excel, etc.),\n
            :param     separator: ',' (comma separator, tab separator, etc.),\n
            :param     problemType: 'classification' (classification or regression, bydefault classification),\n
            :param     imputeMissing: KNNImputer (to handle the outliers by KNNImputer or mean),\n
            :param     outlier_threshold: 3 (set the thereshold value to delete the outliers),\n
            :param     handle_outliers: True (True or False --> True: remove the outliers, False: not to remove outliers),\n
            :param    mapping_ycol: False (True or False --> True: mapping the output column, False: not to map output column),\n
            :param     balance_data: False (True or False --> True: balanced the data, False: not to balance the data0,\n
            :param     dropCols: None (mention columns name for dropping, bydefault no columns drop),\n
            :param     scalingType: None (for data scaling you can use normalized or standarized or quantil transformation, bydefault None),\n
            :param     dataTransformationType: None (for data transformation using log or sqrt or boxcox, bydefault None),\n
            :param     chooseAlgorithm: dt (dt: Decision Tree, select)\n\n

            :param for classification:
            :param dt --> Decision Tree,\n
            :param rf --> Random Forest,\n
            :param xg --> XGBoost,\n
            :param ensemble_dt --> Ensemble Tequnique, base model Decision Tree,\n
            :param ensemble_knn --> Ensemble Tequnique, base model KNN,\n
            :param best_model --> get the best model by comparing all model,

            :param for regression:
            :param linear --> Liner Regression,\n
            :param lasso  --> Lasso Regression,\n
            :param ridge  --> Ridge Regression,\n
            :param elasticnet --> ElasticNet,\n
            :param best_model --> get best model by comparing all model

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file = open(self.file_path, 'a+')
            self.data_path = data_path
            self.ycol = yCol
            self.format = format
            self.separator = separator
            self.imputeMissing = imputeMissing
            self.outlier_threshold = outlier_threshold
            self.handle_outliers = handle_outliers
            self.mapping_ycol = mapping_ycol
            self.balance_data = balance_data
            self.dropCols = dropCols
            self.scalingType = scalingType
            self.dataTransformationType = dataTransformationType
            self.chooseAlgorithm = chooseAlgorithm
            self.problemType = problemType
            self.data = self.data_load.get_data(path=self.data_path, format=self.format, separator=self.separator)
            # validate the data before training for classification and regression:
            self.data = self.validation.ValidateTrainingData(data=self.data, yCol=self.ycol,
                                                             imputeMissing=self.imputeMissing,
                                                             outlier_threshold=self.outlier_threshold,
                                                             handle_outliers=self.handle_outliers,
                                                             mapping_ycol=self.mapping_ycol,
                                                             balance_data=self.balance_data,
                                                             dataTransformationType=self.dataTransformationType)

            # Drop the column or columns based on given condition:
            if self.dropCols is None:
                self.logger_object.log(self.file, "No need to drop the column or columns")
            else:
                self.data = self.pre_processing.ToDroColumns(data=self.data, Xcols=self.dropCols)
                self.logger_object.log(self.file, f"Successfully drop the columns: {self.dropCols}")
            # if the problem statemet is classification:
            if self.problemType.lower() == 'classification':
                """ Perform the data pre-processing """
                # Separate the features columns and label columns (Xcols & Ycols):
                self.X, self.Y = self.pre_processing.ToSeparateTheLabelFeature(data=self.data, Ycol=self.ycol)
                self.logger_object.log(self.file,
                                       f"Successfully separate the labels: xCol {str(self.X.columns)}, yCol: {self.ycol}")
                # Drop those column/columns which has/have zero standard deviation:
                self.zero_std_col = self.pre_processing.ToGetColumnsWithZeroStandardDeviation(data=self.X)
                self.logger_object.log(self.file, f"Get the zero standard deviation columns:  {self.zero_std_col}")
                if len(self.zero_std_col) > 0:
                    self.X = self.pre_processing.ToDroColumns(data=self.X, Xcols=self.zero_std_col)
                    self.logger_object.log(self.file, "Drop those columns which have zero standard deviation")

                """ Aplying KMeans Clustering Approaches """
                # apply elbow method to get the number of cluster:
                self.n_cluster = self.clustering.Elbow_Method(data=self.X)
                self.logger_object.log(self.file, f"Get the number of cluster using Elbow Method  {self.n_cluster}")
                # Devide the data into that number of cluster:
                self.X = self.clustering.ToCreateCluster(data=self.X, no_cluster=self.n_cluster)
                self.logger_object.log(self.file, f"Create the {self.n_cluster} clusters")
                # To label the output features:
                self.X['Labels'] = self.Y
                # getting the unique cluster from the data set:
                self.list_of_cluster = self.X['cluster_label'].unique()
                self.logger_object.log(self.file, f"get the {self.list_of_cluster} list of clusters")

                """parsing all the clusters and looking for the best ML_Algorithm or given ML_Algorithm to fit on individual cluster"""
                # if select the Decesion Tree Classifier:
                if self.chooseAlgorithm.lower() in ['decision tree', 'decisiontree', 'dt']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.cluster_features,
                                                                                                self.cluster_label,
                                                                                                test_size=0.33,
                                                                                                random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using DecisionTree and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyDT(x_train=self.x_train,
                                                                       x_test=self.x_test,
                                                                       y_train=self.y_train,
                                                                       y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"DecisionTree_{str(i)}")
                    self.logger_object.log(self.file,
                                           f"Successfully train and saved the model using {self.chooseAlgorithm} algorithm")
                    self.file.close()
                    return self.model, self.score

                # if select the RandomForest Classifier:
                if self.chooseAlgorithm.lower() in ['random forest', 'randomforest', 'rf']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.cluster_features,
                                                                                                self.cluster_label,
                                                                                                test_size=0.33,
                                                                                                random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using RandomForest and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyRF(x_train=self.x_train,
                                                                       x_test=self.x_test,
                                                                       y_train=self.y_train,
                                                                       y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"RandomForest_{str(i)}")
                    self.logger_object.log(self.file,
                                           f"Successfully train and saved the model {self.chooseAlgorithm} algorithm")
                    self.file.close()
                    return self.model, self.score

                # if select XGBoost:
                if self.chooseAlgorithm.lower() in ['xgboost', 'xg boost', 'xg']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.cluster_features,
                                                                                                self.cluster_label,
                                                                                                test_size=0.33,
                                                                                                random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using XgBoost and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyXGBoost(x_train=self.x_train,
                                                                            x_test=self.x_test,
                                                                            y_train=self.y_train,
                                                                            y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"XGBoost_{str(i)}")
                    self.logger_object.log(self.file,
                                           f"Successfully train and saved the model {self.chooseAlgorithm} algorithm")
                    self.file.close()
                    return self.model, self.score

                # if select the Ensemble approach where base is DecisionTree Classifier:
                if self.chooseAlgorithm.lower() in ['ensemble decision tree', 'ensemble_decision_tree', 'ensemble_dt']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.cluster_features,
                                                                                                self.cluster_label,
                                                                                                test_size=0.33,
                                                                                                random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using Ensemble Technique (base is Decision Tree) and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyEnsembleDT(x_train=self.x_train,
                                                                               x_test=self.x_test,
                                                                               y_train=self.y_train,
                                                                               y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"Ensemble_DT{str(i)}")
                    self.logger_object.log(self.file,
                                           f"Successfully train and saved the model {self.chooseAlgorithm} algorithm")
                    self.file.close()
                    return self.model, self.score

                # if select the Ensemble approach where base is KNN:
                if self.chooseAlgorithm.lower() in ['ensemble knn', 'ensemble_knn',
                                                    'ensembleknn']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using Ensemble Technique (base is KNN) and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyEnsembleKNN(x_train=self.x_train,
                                                                                x_test=self.x_test,
                                                                                y_train=self.y_train,
                                                                                y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"Ensemble_KNN{str(i)}")
                    self.logger_object.log(self.file,
                                           f"Successfully train and saved the model {self.chooseAlgorithm} algorithm")
                    self.file.close()
                    return self.model, self.score

                # if select the best_model:
                if self.chooseAlgorithm.lower() in ['best model', 'best_model',
                                                    'best_md']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)

                        # train the model using DecisionTree and then save the model in 'Model/' directory
                        self.model_name, self.model, self.score = self.model1.ForClassificationALL(x_train=self.x_train,
                                                                                                   x_test=self.x_test,
                                                                                                   y_train=self.y_train,
                                                                                                   y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.best_model,
                                                                        filename=f"{self.model_name}_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

            # if the problem statemet is Regression:
            if self.problemType.lower() == 'regression':
                """ Perform the data pre-processing """
                # Separate the features columns and label columns (Xcols & Ycols):
                self.X, self.Y = self.pre_processing.ToSeparateTheLabelFeature(data=self.data, Ycol=self.ycol)
                self.logger_object.log(self.file,
                                       f"Successfully separate the labels: xCol {str(self.X.columns)}, yCol: {self.ycol}")
                # Drop those column/columns which has/have zero standard deviation:
                self.zero_std_col = self.pre_processing.ToGetColumnsWithZeroStandardDeviation(data=self.X)
                self.logger_object.log(self.file, f"Get the zero standard deviation columns:  {self.zero_std_col}")
                if len(self.zero_std_col) > 0:
                    self.X = self.pre_processing.ToDroColumns(data=self.X, Xcols=self.zero_std_col)
                    self.logger_object.log(self.file, "Drop those columns which have zero standard deviation")

                """ Aplying KMeans Clustering Approaches """
                # apply elbow method to get the number of cluster:
                self.n_cluster = self.clustering.Elbow_Method(data=self.X)
                self.logger_object.log(self.file, f"Get the number of cluster using Elbow Method  {self.n_cluster}")
                # Devide the data into that number of cluster:
                self.X = self.clustering.ToCreateCluster(data=self.X, no_cluster=self.n_cluster)
                self.logger_object.log(self.file, f"Create the {self.n_cluster} clusters")
                # To label the output features:
                self.X['Labels'] = self.Y
                # getting the unique cluster from the data set:
                self.list_of_cluster = self.X['cluster_label'].unique()
                self.logger_object.log(self.file, f"get the {self.list_of_cluster} list of clusters")

                """parsing all the clusters and looking for the best ML_Algorithm or given ML_Algorithm to fit on individual cluster"""
                # if select the Linear Regression:
                if self.chooseAlgorithm.lower() in ['linear model', 'linear regression', 'linearregression', 'linear',
                                                    'linear_regression']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using Linear Regression and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyLinear(x_train=self.x_train,
                                                                           x_test=self.x_test,
                                                                           y_train=self.y_train,
                                                                           y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"LinearRegression_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

                # if select Lasso Regression:
                if self.chooseAlgorithm.lower() in ['lasso model', 'lasso regression', 'lassoregression', 'lasso',
                                                    'lasso_regression']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using Lasso Regression and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyLasso(x_train=self.x_train,
                                                                          x_test=self.x_test,
                                                                          y_train=self.y_train,
                                                                          y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"LassoRegression_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

                # if select Ridge Regression:
                if self.chooseAlgorithm.lower() in ['ridge model', 'ridge regression', 'ridgeregression', 'ridge',
                                                    'ridge_regression']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using Ridge Regression and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyRidge(x_train=self.x_train,
                                                                          x_test=self.x_test,
                                                                          y_train=self.y_train,
                                                                          y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"RidgeRegression_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

                # if select ElasticNet Regression:
                if self.chooseAlgorithm.lower() in ['elasticnet model', 'elasticnet regression', 'elasticnetregression',
                                                    'elasticnet',
                                                    'elasticnet_regression']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using ElasticNet Regression and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlyElasticNet(x_train=self.x_train,
                                                                               x_test=self.x_test,
                                                                               y_train=self.y_train,
                                                                               y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"ElasticNetRegression_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

                # if select SVR regression (Support Vector Regression):
                if self.chooseAlgorithm.lower() in ['svr model', 'svr regression', 'svrregression',
                                                    'svr',
                                                    'svr_regression']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using SVR Regression and then save the model in 'Model/' directory
                        self.model, self.score = self.model1.ForOnlySVR(x_train=self.x_train,
                                                                        x_test=self.x_test,
                                                                        y_train=self.y_train,
                                                                        y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"SVRRegression_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

                # if select the best model:
                if self.chooseAlgorithm.lower() in ['best model', 'best regression', 'bestregression',
                                                    'best',
                                                    'best_regression', 'best_regression_model', 'best_model']:
                    for i in self.list_of_cluster:  # filter the data for one by one cluster
                        self.cluster_data = self.X[self.X['cluster_label'] == i]
                        self.cluster_features = self.cluster_data.drop(axis=1, columns=['Labels',
                                                                                        'cluster_label'])  # separate the features columns
                        self.cluster_label = self.cluster_data['Labels']  # separate the output columns
                        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                            self.cluster_features, self.cluster_label, test_size=0.33, random_state=131)
                        # After splitting apply fiatures Scaling techniques:
                        # Based on condition apply the normalization
                        if self.scalingType is None:
                            pass
                        if self.scalingType.lower() in ['normalized', 'normalize', 'normalization', 'normal']:
                            self.x_train = self.scaling.ToNormalized(data=self.x_train)
                            self.x_test = self.scaling.ToNormalized(data=self.x_test)
                        # Based on condition apply the standarization
                        if self.scalingType.lower() in ['standarized', 'standarize', 'standarization', 'stand']:
                            self.x_train = self.scaling.ToStandarized(data=self.x_train)
                            self.x_test = self.scaling.ToStandarized(data=self.x_test)
                        # Based on condition apply the quantil Transformation
                        if self.scalingType.lower() in ['quantil', 'quantilized', 'quantiltransformation', 'quant',
                                                        'quantil transformation']:
                            self.x_train = self.scaling.ToQuantilTransformerScaler(data=self.x_train)
                            self.x_test = self.scaling.ToQuantilTransformerScaler(data=self.x_test)
                        # train the model using SVR Regression and then save the model in 'Model/' directory
                        self.model_name, self.model, self.score = self.model1.ForRegressionALL(x_train=self.x_train,
                                                                                               x_test=self.x_test,
                                                                                               y_train=self.y_train,
                                                                                               y_test=self.y_test)
                        self.save_model = File_Operations().ToSaveModel(model=self.model,
                                                                        filename=f"{self.model_name}_{str(i)}")
                    self.logger_object.log(self.file, f"Successfully train and save the best model")
                    self.file.close()
                    return self.model, self.score

        except Exception as ex:
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex

    def prediction(self, data, yCol=None, outlier_threshold=3, handle_outliers=False, mapping_ycol=False,
                   imputeMissing='KNNImputer', balance_data=False, scaling=None, dataTransformationType=None):
        """
            Method Name: prediction
            Description: This method helps to predict the outcome based on given data

            :param data: pass the data
            :param yCol: mention the ycol name
            :param outlier_threshold: 3 (bydefault)
            :param handle_outliers: False (True or False: True --> remove the outliers, False --> not remove outliers)
            :param mapping_ycol: False
            :param imputeMissing: KNNImputer (KNNImputer or mean)
            :param balance_data: False (True or False: True --> balance the data, False --> not balance the data)
            :param scaling: None (normalized or standarized or quantil transformation, bydefault None)
            :param dataTransformationType: None (log or sqrt or boxcox)

            Output: predect outcome
            On Failure: Raise Error

            Written By: Dibyendu Biswas
            Version: 1.0
            Revisions: None
        """
        try:
            self.file_path = "Executions_Logs/Predection_logs/Model_Predection_Logs.txt"
            self.file = open(self.file_path, 'a+')
            self.data = data
            self.ycol = yCol
            self.outlier_threshold = outlier_threshold
            self.handle_outliers = handle_outliers
            self.mapping_ycol = mapping_ycol
            self.imputeMissing = imputeMissing
            self.balance_data = balance_data
            self.scaling = scaling
            self.dataTransformationType = dataTransformationType

            # Validate the data before start predection:
            self.data = self.validationP.ValidatePredectionData(data=self.data, yCol=self.ycol,
                                                                outlier_threshold=self.outlier_threshold,
                                                                handle_outliers=self.handle_outliers,
                                                                mapping_ycol=self.mapping_ycol,
                                                                imputeMissing=self.imputeMissing,
                                                                balance_data=self.balance_data,
                                                                dataTransformationType=self.dataTransformationType)
            if self.ycol is None:
                pass
            else:
                self.data = self.data.drop(axis=1, columns=[self.ycol])
            self.file_loader = File_Operations()
            """ Applying the Clustering Approach """
            # to find the cluster, which cluster the data set is belong:
            self.KMeans = self.file_loader.ToLoadModel("KMeans")
            self.cluster_label = self.KMeans.fit_predict(self.data)
            self.data['cluster_label'] = self.cluster_label
            self.no_cluster = self.data['cluster_label'].unique()

            for i in self.no_cluster:
                self.cluster_data = self.data[self.data['cluster_label'] == i]
                self.cluster_data = self.cluster_data.drop(axis=1, columns=['cluster_label'])
                self.model_name = self.file_loader.ToFindCorrectModel(cluster_number=i)
                self.model = self.file_loader.ToLoadModel(filename=self.model_name)
                self.result = list(self.model.predict(self.cluster_data))
                print(self.result)



        except Exception as ex:
            self.file_path = "Executions_Logs/Predection_logs/Model_Predection_Logs.txt"
            self.file = open(self.file_path, 'a+')
            self.logger_object.log(self.file, f"Error is: {ex}")
            self.file.close()
            raise ex


if __name__ == '__main__':
    data = pd.read_csv("Raw Data/diabetes.csv")
    # print(data)
    auto = Automated_ML()
    auto.training(data_path="Raw Data/diabetes.csv",yCol='Outcome',
                  chooseAlgorithm='dt', scalingType='normalized', balance_data=False
                  )
    auto.prediction(data=data, yCol='Outcome', scaling='normalized')




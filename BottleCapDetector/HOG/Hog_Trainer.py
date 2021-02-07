import cv2 as cv
import numpy as np
import os, glob
from sklearn import svm, neighbors, tree , ensemble, naive_bayes 
import pickle
from BottleCapDetector.Helpers.Helper import ClusterImage
from sklearn.metrics import confusion_matrix
import pandas as pd

winSize = (32,32)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog = cv.HOGDescriptor(winSize,
                        blockSize,
                        blockStride,
                        cellSize,
                        nbins,
                        derivAperture,
                        winSigma,
                        histogramNormType,
                        L2HysThreshold,
                        gammaCorrection,
                        nlevels)

winStride = (8,8)
padding = (8,8)
locations = ((10,20),)

class Hog_Model_Trainer:
    def __init__(self, features_directory, model_output_directory='./Models'):
        self.features_directory = features_directory
        self.model_output_directory = model_output_directory

    def TrainMoodel(self):
        image_feature_data, image_label_data = self.GetDataAndLabels(self.features_directory)
        data_quantity = image_feature_data.shape[0]

        random_index_array = np.arange(data_quantity)
        np.random.shuffle(random_index_array)

        image_feature_data = image_feature_data[random_index_array]
        image_label_data = image_label_data[random_index_array]

        train_ratio = 0.8
        slice_point = int(np.floor(data_quantity*train_ratio))
        train_x = image_feature_data[0:slice_point]
        train_y = image_label_data[0:slice_point]
        test_x = image_feature_data[slice_point:]
        test_y = image_label_data[slice_point:]
        
        train_x = self.reshapeDataset(train_x)
        test_x = self.reshapeDataset(test_x)

        models = {}
        models['svm_linear'] = svm.SVC(kernel='linear', C=2, decision_function_shape='ovo').fit(train_x, train_y)
        models['svm_rbf'] = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(train_x, train_y)
        models['svm_poly'] = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(train_x, train_y)
        models['svm_sigmoid'] = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(train_x, train_y)
        models['knn'] = neighbors.KNeighborsClassifier(7).fit(train_x, train_y)
        models['Descision_Tree'] = tree.DecisionTreeClassifier().fit(train_x, train_y)
        models['AdaBoost'] = ensemble.AdaBoostClassifier().fit(train_x, train_y)
        models['Random_Forest'] = ensemble.RandomForestClassifier().fit(train_x, train_y)
        models['Naive_Bayes'] = naive_bayes.MultinomialNB().fit(train_x, train_y)

        accuracies = {}
        for key,value in models.items():
            accuracies[key] = value.score(test_x, test_y)
            # print(value.classes_)
            filename = key + '.sav'
            pickle.dump(value, open(os.path.join(self.model_output_directory,filename), 'wb'))

        for key, value in accuracies.items():
            print('Accuracy for {kernal} is {acc}'.format(kernal = key, acc = value))


        pred_y = models['svm_poly'].predict(test_x)
        self.PrintConfusionMatrix(test_y, pred_y)

    def PrintConfusionMatrix(self, actual_y, predicted_y):
        # print(confusion_matrix(actual_y, predicted_y))
        df_confusion = pd.crosstab(actual_y, predicted_y, rownames=['Actual'], colnames=['Predicted'], margins=True)
        print(df_confusion)


    def reshapeDataset(self, ds):
        # return ds
        nsamples, nx, ny = ds.shape
        return ds.reshape((nsamples,nx*ny))

    def GetDataAndLabels(self, feature_directory):
        image_feature_data = []
        image_label_data = []
        for feature_folder_name in os.listdir(feature_directory):
            files = glob.glob(os.path.join(feature_directory, feature_folder_name, '*.png'))

            for file in files:
                image = cv.imread(file)
                if image.shape[0] < winSize[0] or image.shape[1] < winSize[1]:
                    # print('Incorrect size of file : ' + str(file) + '. Size is : ' + str(image.shape) + '. Expected size is atleast : ' + str(winSize))
                    continue
                
                # histogram = hog.compute(image)
                # image = ClusterImage(image, 2)
                histogram = hog.compute(image,winStride,padding,locations)

                image_feature_data.append(histogram)
                image_label_data.append(feature_folder_name)

        return np.array(image_feature_data), np.array(image_label_data)

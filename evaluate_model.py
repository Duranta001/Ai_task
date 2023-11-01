# import required packages

import cv2 as cv #use for performing various function for image.
import os # use to make connection with operating system.
import numpy as np #use for performing statistical function
import tensorflow as tf #use for deep learning algorithm.
from tensorflow import keras #use for image related deep learning work.
#from tensorflow.keras import layers #use for creating deeplearning layers
#from tensorflow.keras.models import Sequential #use to create sequential model  of deep learning.
from sklearn.metrics import confusion_matrix , classification_report
#import pathlib # use to creat filepath.
import pandas as pd
import sys



try:
    #importing trained model
    model = keras.models.load_model("fasion_model.h5")

    #processing the test data
    x_t = pd.read_csv(input('Enter the test CSV dataset file path: '))
    x_test = x_t.drop('label', axis = 1)
    y_test = x_t['label']
    x_test = np.array(x_test)
    x_test = x_test.reshape(-1,28,28,1)
    x_test_scaled = x_test/255
    
    
    #Model evaluation with test dataset
    result = model.evaluate(x_test_scaled,y_test, verbose=0)
    print("Accuracy: ", result[1])
    
    predicted = model.predict(x_test_scaled,verbose=0)
    predicted1 = np.round(predicted)
    predicted2 = []
    for i in range(len(predicted1)):
        z = np.argmax(predicted1[i])
        predicted2.append(z)
    matrics = classification_report(y_test,predicted2)
    #print(matrics)

    #Evaluate data of a folder
    data_path = input('Enter the folder path: ')
    
    data = os.listdir(data_path)
    if not data:
        raise Exception(f"The folder '{data_path}' does not contain any files.")
    test_data = []
    print("Prediction for all data of the folder: ")
    for image in data:
        img = cv.imread(data_path+'/'+image)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        resized_img = cv.resize(img,(28,28))
        test_data.append(resized_img)
    test_data = np.array(test_data)/255
    test_data = test_data.reshape(-1,28,28,1)
    predicted = model.predict(test_data,verbose=0)
    predicted1 = np.round(predicted)
    label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    Predicted_output = []
    for i in range(len(predicted1)):
        z = np.argmax(predicted1[i])
        Predicted_output.append(label[z])
        print(label[z])
    print("\n Model's architecture summary, Evaluation metric(s), prediction for given folder data, and other insights available in output.txt file")
   
    # Creating a test file which contain output information
    output_file = 'output.txt'
    with open(output_file, 'w') as f:
        f.write("Model's architecture summary: \n")
        sys.stdout = f
        model.summary()
        sys.stdout = sys.__stdout__
        f.write(" \n")
        f.write('=================================================================================\n')
        f.write('Evaluation metric(s) based on test data \n')
        f.write(matrics)
        f.write(" \n")
        f.write('=================================================================================\n')
        f.write('Evaluation of all the data from the given folder serially:')
        for i in Predicted_output:
            f.write(' \n')
            f.write(i)
except FileNotFoundError:
    try:
        print(f"The folder '{data_path}' does not exist.")
    except NameError:
        print("The test data file path is incorrect.")
except OSError:
    print(".h5 model not found or entered a wrong path")
except Exception as e:
    print(str(e))

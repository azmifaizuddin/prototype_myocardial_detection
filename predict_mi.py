#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import io
from PIL import Image
from matplotlib import image as mpimg
from sklearn.metrics import accuracy_score
import pickle

named_labels = ['MI', 'NORM', 'OTHER']
named_labels = np.array(named_labels)


# In[55]:


def predict(data_path, mod):
    # Make predictions on validation set
    with open('modeladam_CNN', 'rb') as f :
        model_cnn = pickle.load(f)
    with open('modeladam_LSTM', 'rb') as f :
        model_lstm = pickle.load(f)
    with open('modeladam_DNN', 'rb') as f :
        model_dnn = pickle.load(f)
    data_test = np.load(data_path)
    label_test = np.load('label_test.npy')
    if mod == 'cnn' :
        model = model_cnn
    elif mod == 'lstm' :
        model = model_lstm
    elif mod == 'dnn' :
        model = model_dnn
    Y_pred = model.predict(data_test)
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    Y_true_labels = np.argmax(label_test, axis=1)
    # Matriks konfusi
    conf_matrix = confusion_matrix(Y_true_labels, Y_pred_labels, labels=np.arange(len(named_labels)))

    # Accuracy Score
    accuracy = accuracy_score(Y_true_labels, Y_pred_labels)
    return conf_matrix, accuracy


def model_performance(data, mod):
    cf_matrix, all_accuracy = predict(data, mod)
    performance_result = []
    mean_performance_result = []
    true_positive = np.diag(cf_matrix)
    false_negative = np.sum(cf_matrix, axis=1) - true_positive
    false_positive = np.sum(cf_matrix, axis=0) - true_positive
    true_negative = np.sum(cf_matrix) - (true_positive + false_negative + false_positive)
    # Calculate the total number of samples
    total_samples = sum(sum(row) for row in cf_matrix)


    accuracy = (true_positive + true_negative) / total_samples
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)

    mean_accuracy = np.mean(accuracy)
    mean_sensitivity = np.mean(sensitivity)
    mean_specificity = np.mean(specificity)

    labels = np.array(['MI', 'NORM', 'OTHER'])
    
    for i in range(len(accuracy)):
        accuracy[i] = round(all_accuracy * 100, 2)
        sensitivity[i] = round(sensitivity[i] * 100, 2)
        specificity[i] = round(specificity[i] * 100, 2)

    formatted_mean_acc = "{:.2f}%".format(all_accuracy*100)
    formatted_mean_sensi = "{:.2f}%".format(mean_sensitivity*100)
    formatted_mean_spec = "{:.2f}%".format(mean_specificity*100)

    performance_result.extend((labels, accuracy, sensitivity,  specificity))
    mean_performance_result.extend((labels, formatted_mean_acc, formatted_mean_sensi,  formatted_mean_spec))
    
    return cf_matrix, performance_result, mean_performance_result



def create_confusion_matrix(cf_matrix, path):
    # Calculate percentages for each true class
    conf_matrix_percentage = cf_matrix
    # Create a heatmap using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.0f', cmap='Blues', xticklabels=named_labels, yticklabels=named_labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(path)

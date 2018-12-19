from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from pandas_ml import ConfusionMatrix

import pandas as pd
import numpy as np
import csv
import math
import pickle
import sklearn

#csv to Pandas DataFrame
def csvToDataframe(filepath):
    return pd.read_csv(filepath)

# Model fitting
def modelFitting(classifier, X_train, y_train):
    model = classifier
    model.fit(X_train, y_train)
    return model

# Model score
def sgdScore(model, X_test, y_test):
    return model.score(X_test, y_test)

# Predicts label based on model 
def sgdPredictions(df, model):
    test_data = pd.DataFrame(createTestData(df))
    test_data.columns = ['user_weight', 'review_count', 'stars_x']

    test_labels = model.predict(np.array(test_data))

    predictions_file = open('SGDClassifier_predictions.csv', 'w')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['user_weight', 'review_count', 'stars_x', 'predicted_is_open'])
    open_file_object.writerows(zip(test_data['user_weight'], test_data['review_count'], test_data['stars_x'], test_labels))
    predictions_file.close()

# Creates the training list with multiple parameters
# Parameters:
#   df - Dataframe
#   columns: name of all columns you want to add to the list
#
def createTrainingList(df, *columns):
    cols = []
    for col in columns:
        cols.append(df[col])
    return list(zip(*cols))

# Creates the label list
def createLabelList(df, label):
    return list(df[label])

# Export model
def exportModel(name, model):
    pickle.dump(model, open(name + '.pickle', 'wb'))

# Confusion Matrix
def confusionMatrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def main():
    print('Creating DataFrame...')
    df = csvToDataframe('../../../downloads/megakevin2.csv')

    print('Filling null values...')
    df['user_weight'].fillna(0, inplace=True)

    print('Creating new DataFrame...')
    X = createTrainingList(df, 'user_weight', 'review_count', 'stars_x')
    y = createLabelList(df, 'is_open')

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.bool_)

    print('Splitting Data...')
    # Splits X and Y training, and X and Y testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale data to be between -1 and 1 for optimal training
    print('Scaling Data...')
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Define Classifiers
    # Classifier List: SGD, SVC, Linear Regression, Logisitic Regression, 
    # Decision Tree, Random Forest, Nearest Neighbors, Naive Bayes
    classifiers = [
        SGDClassifier(loss='hinge'),
        SVC(kernel='poly', max_iter=100, random_state=42), 
        LinearRegression(), LogisticRegression(random_state=0),
        DecisionTreeClassifier(random_state=0), RandomForestClassifier(random_state=0),
        GaussianNB()
    ]

    print('Training data and scoring...')
    classifier_scores = []
    trained_models = []

    for classifier in classifiers:
        model = modelFitting(classifier, X_train, y_train)
        trained_models.append(model)
        score = (model.score(X_test, y_test))
        classifier_scores.append(score)
        print('Model: ' + str(classifier) + '\nScore: ' + str(round(score * 100, 4)) + '%') 

    print('Max score: ' + str(max(classifier_scores) * 100) + '%')
    
    print('Printing Confusion Matrices')
    for model in trained_models:
        y_pred = model.predict(X_test)
        print(model)
        confusion_matrix = ConfusionMatrix(y_test, y_pred)
        print('Confusion Matrix:\n%s' % confusion_matrix)

if __name__ == '__main__':
    main()
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import csv
import math
import pickle

def csvToDataframe(filepath):
    return pd.read_csv(filepath)

def sgdFitting(X, Y, n_iter):
    clf = SGDClassifier(loss="hinge", max_iter=n_iter, shuffle=True)
    clf.fit(X, Y)
    return clf    

# X array for clf.fit (80% of data)
def createTrainingChunks(df):
    return list(zip(df['user_weight'], df['review_count'], df['stars_x']))

# Y array for clf.fit (80% of data)
def createLabels(df, label):
    return list(df[label])

# Predicts outcome and writes predictions to csv file
def sgdPredictions(df, model):
    test_data = pd.DataFrame(createTestData(df))
    test_data.columns = ['user_weight', 'review_count', 'stars_x']

    test_labels = model.predict(np.array(test_data))

    predictions_file = open('SGDClassifier_predictions.csv', 'w')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['user_weight', 'review_count', 'stars_x', 'predicted_is_open'])
    open_file_object.writerows(zip(test_data['user_weight'], test_data['review_count'], test_data['stars_x'], test_labels))
    predictions_file.close()

# Model score
def sgdScore(model, X_test, y_test):
    return model.score(X_test, y_test)

# Export model
def exportModel(model):
    pickle.dump(model, open('SGDClassifierModel.pickle', 'wb'))

def main():
    print('Reading CSV: megakevin')
    df = csvToDataframe('../../../downloads/megakevin2.csv')
    print('Filling extra data...')

    df['user_weight'].fillna(0, inplace=True)

    print('Splitting data...')

    X = createTrainingChunks(df)
    y = createLabels(df)

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.bool_)

    # Splits X and Y training, and X and Y testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale data to be between -1 and 1 for optimal training
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print('Training data and scoring...')
    iterations =  list(range(1, 100)) + [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    scores = []
    for i in iterations:
        model = sgdFitting(X_train, y_train, i)
        score = (model.score(X_test, y_test))
        scores.append(score)
        print('Iteration: ' + str(i) + '\nScore: ' + str(round(score * 100, 4)) + '%')       
    print('Max score: ' + str(max(scores)) + '\nIteration: ' + str(scores.index(max(scores)) - 1))
    
    print('Plotting scores...')
    plt.title('SGDClassifier n_iter')
    plt.xlabel('n_iter')
    plt.ylabel('score')
    plt.plot(range(1, 1000), scores)
    
    losses = ['hinge', 'log', 'modified_huber', 'perceptron', 'squared_hinge']
    loss_scores = []
    for loss in losses:
        model = SGDClassifier(loss=loss, penalty='12', max_iter=max(scores))
        model.fit(X_train, y_train)
        loss_scores.append(model.score(X_test, y_test))

    print('Plotting scores...')
    plt.title('SGDClassifier loss')
    plt.xlabel('loss')
    plt.ylabel('score')
    x = np.arange(len(losses))
    plt.xticks(x, losses)
    plt.plot(x, loss_scores)
    
    print('Exporting model...')
    exportModel(trained)

if __name__ == '__main__':
    main()
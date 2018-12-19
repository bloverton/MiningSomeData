from sklearn.kernel_approximation import RBFSampler

def csvToDataframe(filepath):
    return pd.read_csv(filepath)

def featureSampler(X):
    rbf_feature = RBFSampler(gamma=1, random_state=1)
    X_features = rbf_feature.fit_transform(X)
    return X_features

def main():
    print('Starting Kernel Approximation')
    print('Converting CSV to Dataframe...')
    df = csvToDataframe('../../../downloads/megakevin2.csv')
    #X = df.<some x value>
    print('Creating kernel map...')
    X_features = featureSampler(X)

    print('Training model...')
    model = SGDClassifier(X_features, Y)

    print('Scoring test data...')
    SGDClassiferScore(model, X, Y)

if __name__ == __main__:
    main()
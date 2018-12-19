import time
import pandas as pd
import matplotlib.pyplot as plt

minReviewCountAtEliteVal = []
minComplimentAtEliteVal = []
minMiscScoreAtEliteVal = []

maxReviewCountAtEliteVal = []
maxComplimentAtEliteVal = []
maxMiscScoreAtEliteVal = []

def csvToDataframe(fileName):
    return pd.read_csv(fileName)

def createScatterPlot(df, param1, param2):
    # Create Data
    N = len(df)
    x = []
    y = []
    for var in range(0, N):
        x.append(df.loc[var, param1])
        y.append(df.loc[var, param2])
    
    colors = (0, 0, 0)
    
    #Create Plot
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.title('Scatter plot ' + param1 + ' vs. ' + param2)
    plt.xlabel(param1)
    plt.ylabel(param2)

def showGraphs():
    plt.show()

def dataframeToCsv(df):
    df.to_csv('WeightedUserDataset', index=False)

def minMaxCounts(df):
    for i in range (0, 15):
        minReviewCountAtEliteVal.append(df.loc[df['elite'] == i, 'review_count'].min())
        minComplimentAtEliteVal.append(df.loc[df['elite'] == i, 'compliment'].min())
        minMiscScoreAtEliteVal.append(df.loc[df['elite'] == i, 'misc_score'].min())

        maxReviewCountAtEliteVal.append(df.loc[df['elite'] == i, 'review_count'].max())
        maxComplimentAtEliteVal.append(df.loc[df['elite'] == i, 'compliment'].max())
        maxMiscScoreAtEliteVal.append(df.loc[df['elite'] == i, 'misc_score'].max())   

def calcUserWeight(elite_val, average_stars, review_count, compliment, misc_score):

    #calculate weight
    #*weight needs to be normalized for a score from 0-1*
    #weight should have a normal distribution
    weight = (average_stars / 5) * (
        (elite_val / 14) + (
            ((review_count - minReviewCountAtEliteVal[elite_val]) / (maxReviewCountAtEliteVal[elite_val] - minReviewCountAtEliteVal[elite_val])) +
            ((compliment - minComplimentAtEliteVal[elite_val]) / (maxComplimentAtEliteVal[elite_val] - minComplimentAtEliteVal[elite_val])) +
            ((misc_score - minMiscScoreAtEliteVal[elite_val]) / (maxMiscScoreAtEliteVal[elite_val] - minMiscScoreAtEliteVal[elite_val]))
        )
    )

    return weight

def createWeightColumn(df):
    userWeights = []
    N = len(df)

    for var in range(0, N):
        elite_val = df.loc[var, 'elite']
        average_stars = df.loc[var, 'average_stars']
        review_count = df.loc[var, 'review_count']
        compliment = df.loc[var, 'compliment']
        misc_score = df.loc[var, 'misc_score']

        userWeights.append(calcUserWeight(elite_val, average_stars, review_count, compliment, misc_score))

    df['user_weight'] = userWeights

    return df

def main():

    start = time.time()

    print('Creating dataframe...')
    df = csvToDataframe('../../../downloads/megakevin2.csv')

    #print('Initializing min and max arrays...')
    #minMaxCounts(df)

    #print('Calculating User Weights... *This may take a while*')
    #newDf = createWeightColumn(df)

    #print('New dataframe with weights')
    #print(newDf.head())

    #print('Creating Scatter Plot: user_weight vs average_stars')
    #createScatterPlot(newDf, 'average_stars', 'user_weight')

    #print('Creating Scatter Plot: elite vs average_stars')
    #createScatterPlot(newDf, 'elite', 'user_weight')

    #showGraphs()

    endtime = time.time() - start
    print("Elapsed time: %.2f" % (endtime / 60) + " minutes")

if __name__ == '__main__': 
    main()

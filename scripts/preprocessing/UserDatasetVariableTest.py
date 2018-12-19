import pandas as pd
import matplotlib.pyplot as plt

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
    plt.show()

def main():
    print('Creating dataframe...')
    df = csvToDataframe('BENBIGBOOBS_ver3.csv')

    print('Creating Scatter Plot: review_count vs elite')
    createScatterPlot(df, 'review_count', 'elite')

    print('Creating Scatter Plot: misc_score vs elite')
    createScatterPlot(df, 'misc_score', 'elite')

    print('Creating Scatter Plot: average_stars vs elite')
    createScatterPlot(df, 'average_stars', 'elite')

    print('Creating Scatter Plot: compliment vs elite')
    createScatterPlot(df, 'compliment', 'elite')

    print('Creating Scatter Plot: review_count vs average stars')
    createScatterPlot(df, 'average_stars', 'review_count')
    
    createScatterPlot(df, 'average_stars', 'elite')

if __name__ == '__main__': 
    main()
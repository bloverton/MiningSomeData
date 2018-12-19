import pandas as pd 
import numpy as np

def main():

    path = 'BENBIGBOOBS_ver3.csv'
    df = pd.read_csv(path)
    temp = 0

    for x in range(0, 1518169):
        #if df.loc[x, 'elite'] == 14:# and temp<df.loc[x,'misc_score']:
        temp = df.loc[x, 'misc_score']
        print(df.loc[x])

    print(temp)

if __name__ == '__main__':
  main()
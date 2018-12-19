import time

import json 
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print('Starting review_count normalization')

    df = pd.read_csv("optimized_yelp_academic_dataset_user1.csv")
    print("Finished Reading .csv file")
    #print(df.info(memory_usage='deep'))
    print("Processing min-max algorithm")
    df['review_count'] = get_max_min(df['review_count'])
    print("Finished min-max algorithm")
    print('Starting df -> csv conversion')
    df.to_csv('yelp_user_dataset_normalized.csv', index=False)

    print('\nFinished Data Transformation')
    print(df.head())
    

def get_max_min(df):
    max = df.max()
    min = df.min()
    df = df.map(lambda x: (x - min / (max - min)))

    return df

def json_to_csv():
    df = pd.read_json("optimized_yelp_academic_dataset_user.json", orient="columns", lines=True)
    df.to_csv('optimized_yelp_academic_dataset_user1.csv', index=False)

if __name__ == "__main__":
    start = time.time()
    main()
    print(time.time() - start)
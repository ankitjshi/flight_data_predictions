import sys

import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=sys.maxsize)

def preprocess(dfm):

    dfm = dfm.drop(['Unnamed: 0'], axis=1)


    print(dfm['UniqueCarrier'].value_counts())
    print(dfm['UniqueCarrier'].shape)

    print(dfm['Dest'].value_counts())
    print(dfm['Dest'].shape)
    dfm = dfm[(dfm['Cancelled'] == 0)]

    plt.figure(figsize=(20, 14))
    ax = sns.heatmap(dfm.corr(), cmap='viridis', center=0, annot=True)
    bottom, top = ax.get_ylim()
    plt.title("Correlation amongst predictors", fontsize=20, color='Black', fontstyle='normal')
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.yticks(rotation=0, fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.show()

    dfm = dfm.drop(['Cancelled', 'Year', 'DayofMonth', 'TailNum', 'CRSDepTime', 'DepTime',
                    'CRSArrTime', 'ArrTime', 'Diverted', 'Cancelled', 'CancellationCode', 'FlightNum', 'AirTime',
                    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'TaxiIn',
                    'TaxiOut','ActualElapsedTime', 'CRSElapsedTime'], axis=1)

    status = []

    for value in dfm['ArrDelay']:
        if value < 15:
            status.append(0)
        else:
            status.append(1)
    dfm['FlightDelayStatus'] = status

    dfm = dfm.drop(['ArrDelay'], axis=1)
    print(dfm.shape)
    dfm = dfm.dropna()
    print(dfm.isnull().sum())
    print(dfm.info())

    dfm = dfm[dfm['Distance'] < 1800]
    dfm = dfm[dfm['DepDelay'] < 100]

    print(dfm.shape)

    # Creating plot
    plt.boxplot(dfm['Distance'])
    plt.show()

    plt.boxplot(dfm['DepDelay'])
    plt.show()

    print("Boxplot Generated")
    print(dfm['FlightDelayStatus'].value_counts())
    return dfm

flights = pd.read_csv('flights_2008.csv')

print(flights['UniqueCarrier'].value_counts())
print(flights['UniqueCarrier'].shape)

print(flights['Cancelled'].value_counts())
print(flights['Cancelled'].shape)

print(flights['Origin'].value_counts())
print(flights['Origin'].shape)

print(flights['Dest'].value_counts())
print(flights['Dest'].shape)

print(flights['Diverted'].value_counts())
print(flights['Diverted'].shape)

airlines_delay = flights.groupby('UniqueCarrier')['ArrDelay'].sum().reset_index(name='num_delays')
# airlines_delay = airlines_delay.sort_values(by='num_delays',ascending=False)
top_airlines_withdelays = airlines_delay.iloc[0:3].rename_axis('num_delays')
airline_list_delay = top_airlines_withdelays['UniqueCarrier'].tolist()
boolean_series = flights.UniqueCarrier.isin(airline_list_delay)
dfm_airline_delay = flights[boolean_series]
# top_airlines_withdelays.plot(kind="bar")

print("Top Airlines")
print(top_airlines_withdelays)

print("*************Post ******")

dfm_airline_delay = preprocess(dfm_airline_delay)
dfm_airline_delay.to_csv('data_cleaned_airlines_with_delay_sorted.csv')

transformed_data = pd.get_dummies(dfm_airline_delay, columns=['Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
                         drop_first=True)
print(transformed_data.head())
print(transformed_data.shape)

transformed_data.to_csv('pre_processed_data_airline_na_dropped.csv')
print("Process Completed")
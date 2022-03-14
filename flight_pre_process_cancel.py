import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import sys
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
np.set_printoptions(threshold=sys.maxsize)

def preprocess(dfm):
    plt.figure(figsize=(20, 14))
    ax = sns.heatmap(dfm.corr(), cmap='viridis', center=0, annot=True)
    bottom, top = ax.get_ylim()
    plt.text(0, -0.6, "df2 - Heat Map", fontsize=30, color='Black', fontstyle='normal')
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.yticks(rotation=0, fontsize=14)
    plt.xticks(rotation=90, fontsize=14)
    plt.show()

    dfm = dfm.drop([ 'Year', 'DayofMonth', 'TailNum', 'CRSDepTime', 'DepTime','Diverted',
                    'CRSArrTime', 'ArrTime', 'FlightNum', 'AirTime','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 'TaxiIn',
                    'TaxiOut','ActualElapsedTime', 'CRSElapsedTime','ArrDelay'], axis=1)
    dfm = dfm.loc[:, ~dfm.columns.str.contains('^Unnamed')]
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
    return dfm

flights = pd.read_csv('flights_2008.csv')
airlines_cancel = flights.groupby('UniqueCarrier')['Cancelled'].sum().reset_index(name='cancelled')
# airlines_cancel = airlines_cancel.sort_values(by='cancelled',ascending=False)
top_airlines_withcancel = airlines_cancel.iloc[0:3].rename_axis('cancelled')
airline_list_cancel = top_airlines_withcancel['UniqueCarrier'].tolist()
boolean_series = flights.UniqueCarrier.isin(airline_list_cancel)
dfm_airline_cancel = flights[boolean_series]
top_airlines_withcancel.plot(kind="bar")

print(dfm_airline_cancel['UniqueCarrier'].value_counts())
print(dfm_airline_cancel['UniqueCarrier'].shape)

print(dfm_airline_cancel['Cancelled'].value_counts())
print(dfm_airline_cancel['Cancelled'].shape)

print(dfm_airline_cancel['Diverted'].value_counts())
print(dfm_airline_cancel['Diverted'].shape)

print("3 Airlines data sorted")
print(top_airlines_withcancel)

print("*************Post ******")

dfm_airline_cancel = preprocess(dfm_airline_cancel)
dfm_airline_cancel.to_csv('data_cleaned_airlines_with_cancel.csv')

transformed_data = pd.get_dummies(dfm_airline_cancel,
                         columns=['CancellationCode', 'Month', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'],
                         drop_first=True)
# transformed_data = transformed_data.dropna()
print(transformed_data.head())
transformed_data.to_csv('pre_processed_cancel_airline_na_dropped.csv')
print(transformed_data.shape)

print("Process Completed")

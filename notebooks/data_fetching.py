import os
import time
from dateutil.parser import parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date


#! Function to fetch data for new company from yahoo finance
def YahooFinanceHistory(company, previous_days):
    '''
    
    This function takes the company name and the number of previous days as input and returns the dataframe of the company history.

    Variables:

    company: string, name of the company
    previous_days: int, number of days to extract data from
    today: date, today's date
    past: date, date of the past
    query_string: string, query string to extract data from yahoo finance
    company_prices: dataframe, dataframe containing the prices of the company
    company_data: dataframe, dataframe containing the data of the company
    valuation_measures: list, list containing the valuation measures interested in
    company_valuation: dataframe, dataframe containing the valuation measures of the company
    path_save_as_csv: boolean, True if the dataframe is to be saved as a csv file, False otherwise
    
    '''
    
    today = int(time.mktime((datetime.now()).timetuple()))
    past = int(time.mktime((datetime.now() - timedelta(previous_days)).timetuple()))
    
    interval = '1d'

    # defining the query to get historical stock data
    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={today}&interval={interval}&events=history&includeAdjustedClose=true'
    
    company_prices = pd.read_csv(query_string)  

    company_prices['Company'] = company
    training_data = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')

    training_data = training_data.append(company_prices)

    training_data1 = training_data[training_data['Company'] == company]
    training_data = training_data[training_data['Company'] != company]

    if training_data1['Date'].tail(1).values[0] != company_prices['Date'].tail(1).values[0]: 
        training_data1 = training_data1.append(company_prices.tail(1))
    else:
        pass

    training_data = training_data.append(training_data1)
    data = training_data[training_data['Company'] == company]
    data1 = training_data[training_data['Company'] != company]
    data.drop_duplicates(subset = 'Date', inplace = True, keep = 'last')
    data.reset_index(inplace = True, drop = True)
    training_data = data1.append(data)
    training_data.to_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv', index = False)

    return company_prices


#! Function to read data from csv file
def read_data(company, previous_days, holidays_list_path = 0):

    company_prices = YahooFinanceHistory(company, previous_days)
    company_prices = company_prices[:-1]
    company_prices = company_prices[['Date', 'Close']]
    company_prices.columns = ['ds', 'y']
    company_prices['ds'] = pd.to_datetime(company_prices['ds'])
    
    if holidays_list_path.endswith('.xls') == True:
        holidays_list = pd.read_excel(holidays_list_path)

    elif holidays_list_path.endswith('.csv') == True:
        holidays_list = pd.read_csv(holidays_list_path)

    elif holidays_list_path.endswith('.xlsx') == True:
        holidays_list = pd.read_excel(holidays_list_path)

    else:
        print('File type not supported')

    for i in range(len(holidays_list['Day'])):
        holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))

    holidays_list = holidays_list[['Holiday','Day']]
    holidays_list = holidays_list.rename({'Day':'ds', 'Holiday':'holiday'}, axis = 1)   

    return company_prices, holidays_list

def fetch_data_new_company(new_company):
    new_company = ','.join(new_company)
    new_company_prices = read_data(new_company, 365*5, '/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv') # read data for 5 years
    return new_company_prices

def data_delete_old_company(old_company):
    old_company = ','.join(old_company)
    training_data = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')
    training_data = training_data[training_data['Company'] != old_company]
    training_data.to_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv', index=False)
    os.remove(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{old_company}.csv')
    os.remove(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{old_company}.json')
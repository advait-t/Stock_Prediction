import json
import time
import prophet
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from dateutil.parser import parse
from datetime import datetime, timedelta, date
from prophet.serialize import model_to_json, model_from_json

from dateutil.parser import parse
from prophet import Prophet

# #! Loading Model
# def load_model(model_path):
#     with open(model_path, 'r') as fin:
#         saved_model = model_from_json(json.load(fin))  # Load model
#     return saved_model

# #! check if today is holiday
# def is_holiday(today):
#     # holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
#     holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
#     for i in range(len(holidays_list['Day'])):
#         holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))
#     for i in range(len(holidays_list['Day'])):
#         if holidays_list['Day'][i].date() == today:
#             return True
#     return False

 #! for next day prediction
# def next_day_prediction(model_path, missing_dates, missing_dates_df = 0):

#     saved_model = load_model(model_path)

#     if missing_dates == False:
#         next_day = date.today() + timedelta(days=1)
#         future_date = pd.DataFrame(pd.date_range(start = next_day, end = next_day, freq ='D'), columns = ['ds'])
#         predicted = saved_model.predict(future_date)
#         return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])
#     else:
#         missing_dates_df.rename(columns={'Date':'ds'}, inplace=True)
#         predicted = saved_model.predict(missing_dates_df)
#         return (predicted[['ds','yhat', 'yhat_upper', 'yhat_lower']])

#! Functions for inferencing
# def real_stock_price(company, predicted):

#     now = datetime.now()
#     weekday_weekend = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S')
    
#     if weekday_weekend.weekday() <= 5 and weekday_weekend.weekday() != 0:
#         days = 1
#     elif weekday_weekend.weekday() == 6:
#         days = 2
#     elif weekday_weekend.weekday() == 0:
#         days = 3

#     past = datetime.strptime(str(predicted['ds'][0]), '%Y-%m-%d %H:%M:%S') - timedelta(days)
#     past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
#     past = int(time.mktime(past.timetuple()))
    
#     interval = '1d'

#     # defining the query to get historical stock data
#     query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
    
#     try:
#         company_stock_price = pd.read_csv(query_string)
#         company_stock_price = company_stock_price[['Date', 'Close']]
#         return company_stock_price
#     except:
#         company_stock_price = pd.DataFrame(np.nan, index = [0], columns=['Date'])
#         return company_stock_price


def check_error_df(error_df):
    # get todays date
    Date = date.today()
    # check if the last date matches today
    if error_df.iloc[-1]['ds'] != Date:
        # get the range of dates from last day to todays date
        date_range = pd.date_range(start = error_df.iloc[-1]['ds'], end = Date, freq ='D')
        # create a dataframe with the range of dates
        # keep only dates that are weekdays
        date_range = pd.DataFrame(date_range, columns = ['Date'])
        date_range = date_range[date_range.weekday() < 5]
        # append the dataframe to the error_df
        error_df = error_df.append(date_range)
        # sort the error_df by date
        error_df = error_df.sort_values(by = 'Date')
        # make predictions for the new dates

    else:
        return False


# def pred_vs_real_comparision(real_stock_price, predicted, error_df, company):

#     df = pd.DataFrame([[np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]], columns=error_df.columns)
#     error_df = pd.concat([error_df, df], ignore_index =True)

#     error_df['Date'].iloc[-1] = str(predicted['ds'].iloc[-1].strftime('%Y-%m-%d'))
#     error_df['Date'] = pd.to_datetime(error_df['Date'])
#     error_df = error_df.set_index('Date')

#     error_df['Predicted_Close'].loc[predicted['ds']] = predicted['yhat'].iloc[-1]
#     error_df['Predicted_Close_Minimum'].loc[predicted['ds']] = predicted['yhat_lower'].iloc[-1]
#     error_df['Predicted_Close_Maximum'].loc[predicted['ds']] = predicted['yhat_upper'].iloc[-1]
    
#     # add compnay name to the dataframe
#     error_df['Company'] = company

#     error_df.insert(0, 'Date', error_df.index)

#     if pd.isna(real_stock_price['Date'])[0] == False:
#         if predicted['ds'].iloc[-1].weekday() == 0:
#             days = 3 #default days = 1
#         elif predicted['ds'].iloc[-1].weekday() == 6:
#             days = 2
#         else:
#             days = 1
            
#         error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] = real_stock_price['Close'].iloc[-1]
#         percent_change = ((error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)] - error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)])/error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)]*100)
#         error_df['Percent_Change_from_Close'].loc[predicted['ds']-timedelta(days)] = percent_change

#         up_or_down_original = error_df['Actual_Close'].loc[predicted['ds']][0]-error_df['Actual_Close'].loc[predicted['ds']-timedelta(days)][0]

#         if up_or_down_original > 0:
#             error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Up'

#         elif up_or_down_original == 0:
#             error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Same'

#         else:
#             error_df['Actual_Up_Down'].loc[predicted['ds']] = 'Down'


#         up_or_down_predicted = error_df['Predicted_Close'].loc[predicted['ds']][0]-error_df['Predicted_Close'].loc[predicted['ds']-timedelta(days)][0]

#         if up_or_down_predicted > 0:
#             error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Up'

#         elif up_or_down_predicted == 0:
#             error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Same'

#         else:
#             error_df['Predicted_Up_Down'].loc[predicted['ds']] = 'Down'
        

#         error_df = error_df[~error_df.index.duplicated(keep='first')]

#     else:
#         pass

#     return error_df

# def real_stock_price_missing_date(company, predicted):
#     now = datetime.now()
#     predicted['Close'] = None
#     for i in range(len(predicted['ds'])):
#         past = datetime.strptime(str(predicted['ds'][i]), '%Y-%m-%d %H:%M:%S')
#         past = past.replace(hour = now.hour, minute = now.minute, second = now.second, microsecond = now.second)
#         past = int(time.mktime(past.timetuple()))
#         interval = '1d'
#         query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={past}&interval={interval}&events=history&includeAdjustedClose=true'
#         company_stock_price = pd.read_csv(query_string)
#         company_stock_price = company_stock_price[['Date', 'Close']]
#         predicted['Close'][i] = company_stock_price['Close'].values[0]
#     return predicted


# def filling_missing_dates(error_df, company):
#     Date = date.today()
    
#     date_range = pd.date_range(start = error_df.iloc[-1]['Date'], end = Date, freq ='B')

#     date_range_df = pd.DataFrame(columns = error_df.columns)
#     date_range_df['Date'] = date_range
#     date_range_df['Date'] = date_range_df['Date'].dt.date

#     for i in range(len(date_range_df['Date'])):
#         if is_holiday(date_range_df['Date'][i]) == True:
#             date_range_df = date_range_df[date_range_df['Date'] != date_range_df['Date'][i]]
            
#     missing_dates_df = next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json',True, date_range_df)
#     missing_dates_df = real_stock_price_missing_date(company, missing_dates_df)

#     # convert ds from datetime to date
#     missing_dates_df['ds'] = missing_dates_df['ds'].dt.date

#     missing_dates_df.rename(columns = {'ds':'Date', 'Close':'Actual_Close', 'yhat':'Predicted_Close', 'yhat_upper':'Predicted_Close_Maximum', 'yhat_lower':'Predicted_Close_Minimum'}, inplace = True)
#     missing_dates_df['Percent_Change_from_Close'] = ((missing_dates_df['Actual_Close'] - missing_dates_df['Predicted_Close'])/missing_dates_df['Actual_Close'])*100

#     missing_dates_df['Actual_Up_Down'] = np.where((missing_dates_df['Actual_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')
#     missing_dates_df['Predicted_Up_Down'] = np.where((missing_dates_df['Predicted_Close'] > missing_dates_df['Actual_Close'].shift(-1)), 'Up', 'Down')

#     error_df = error_df.append(missing_dates_df, ignore_index= True)
#     error_df = error_df.drop_duplicates(subset = 'Date', keep = 'last')
#     error_df['Company'] = company

#     error_df['Actual_Close'] = error_df['Actual_Close'].astype(float)
#     error_df['Predicted_Close'] = error_df['Predicted_Close'].astype(float)
#     error_df['Predicted_Close_Minimum'] = error_df['Predicted_Close_Minimum'].astype(float)
#     error_df['Predicted_Close_Maximum'] = error_df['Predicted_Close_Maximum'].astype(float)
#     error_df['Percent_Change_from_Close'] = error_df['Percent_Change_from_Close'].astype(float)
#     return error_df


# def print_metrics_columns(error_df, Date):
#     actual_close = (error_df[error_df['Date'] == Date]['Actual_Close'].values[0]).round(2)
#     predicted_close = (error_df[error_df['Date'] == Date]['Predicted_Close'].values[0]).round(2)
#     actual_up_down = error_df[error_df['Date'] == Date]['Actual_Up_Down'].values[0]
#     predicted_up_down = error_df[error_df['Date'] == Date]['Predicted_Up_Down'].values[0]
#     if predicted_up_down == 'Up':
#         predicted_up_down =  '+Up'
#     elif predicted_up_down == 'Down':
#         predicted_up_down = '-Down'
    
#     if actual_up_down == 'Up':
#         actual_up_down =  '+Up'
#     elif actual_up_down == 'Down':
#         actual_up_down = '-Down'

#     return actual_close, predicted_close, actual_up_down, predicted_up_down

# def compute_accuracy(y_true, y_pred):
#     correct_predictions = 0
#     # iterate over each label and check
#     for true, predicted in zip(y_true, y_pred):
#         if true == predicted:
#             correct_predictions += 1
#     # compute the accuracy
#     accuracy = round(((correct_predictions/len(y_true))*100),2)

#     return accuracy


# def read_data(company, previous_days, holidays_list_path = 0):

#     company_prices = YahooFinanceHistory(company, previous_days)
#     company_prices = company_prices[:-1]
#     company_prices = company_prices[['Date', 'Close']]
#     company_prices.columns = ['ds', 'y']
#     company_prices['ds'] = pd.to_datetime(company_prices['ds'])
    
#     if holidays_list_path.endswith('.xls') == True:
#         holidays_list = pd.read_excel(holidays_list_path)

#     elif holidays_list_path.endswith('.csv') == True:
#         holidays_list = pd.read_csv(holidays_list_path)

#     elif holidays_list_path.endswith('.xlsx') == True:
#         holidays_list = pd.read_excel(holidays_list_path)

#     else:
#         print('File type not supported')

#     for i in range(len(holidays_list['Day'])):
#         holidays_list['Day'][i] = pd.to_datetime(parse(holidays_list['Day'][i]))

#     holidays_list = holidays_list[['Holiday','Day']]
#     holidays_list = holidays_list.rename({'Day':'ds', 'Holiday':'holiday'}, axis = 1)   

#     return company_prices, holidays_list


# Function to extract the data from yahoo finance

# def YahooFinanceHistory(company, previous_days):
#     '''
    
#     This function takes the company name and the number of previous days as input and returns the dataframe of the company history.

#     Variables:

#     company: string, name of the company
#     previous_days: int, number of days to extract data from
#     today: date, today's date
#     past: date, date of the past
#     query_string: string, query string to extract data from yahoo finance
#     company_prices: dataframe, dataframe containing the prices of the company
#     company_data: dataframe, dataframe containing the data of the company
#     valuation_measures: list, list containing the valuation measures interested in
#     company_valuation: dataframe, dataframe containing the valuation measures of the company
#     path_save_as_csv: boolean, True if the dataframe is to be saved as a csv file, False otherwise
    
#     '''
    
#     today = int(time.mktime((datetime.now()).timetuple()))
#     past = int(time.mktime((datetime.now() - timedelta(previous_days)).timetuple()))
    
#     interval = '1d'

#     # defining the query to get historical stock data
#     query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1={past}&period2={today}&interval={interval}&events=history&includeAdjustedClose=true'
    
#     company_prices = pd.read_csv(query_string)  

#     company_prices['Company'] = company
#     training_data = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')

#     training_data = training_data.append(company_prices)

#     training_data1 = training_data[training_data['Company'] == company]
#     training_data = training_data[training_data['Company'] != company]

#     if training_data1['Date'].tail(1).values[0] != company_prices['Date'].tail(1).values[0]: 
#         training_data1 = training_data1.append(company_prices.tail(1))
#     else:
#         pass

#     training_data = training_data.append(training_data1)
#     data = training_data[training_data['Company'] == company]
#     data1 = training_data[training_data['Company'] != company]
#     data.drop_duplicates(subset = 'Date', inplace = True, keep = 'last')
#     data.reset_index(inplace = True, drop = True)
#     training_data = data1.append(data)
#     training_data.to_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv', index = False)

#     return company_prices



# def model_building_prophet(company, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = False):

# # defining the hyperparameters for the model building
#     if holidays_list is not None:

#         # variables for the model building and their meaning:
#         '''
#         holidays: list, list of holidays
#         n_changepoints: int, number of changepoints. Change points are abrupt variations in time series data. (n_changepoints = 1 means there is only one changepoint.)
#         n_changepoints_scale: float, scale of the number of changepoints 
#         changepoint_prior_scale: float, scale of the changepoint prior
#         yearly_seasonality: boolean, True if yearly seasonality is to be used, False otherwise
#         weekly_seasonality: boolean, True if weekly seasonality is to be used, False otherwise
#         daily_seasonality: boolean, True if daily seasonality is to be used, False otherwise
#         holidays_prior_scale: float, scale of the holiday prior
#         holidays_yearly_prior_scale: float, scale of the yearly holiday prior
#         fourier_order: int, order of the fourier series. How quickly the seasonility of the time series can change.
#         '''

#         m = Prophet(growth="linear",
#             holidays= holidays_list,
#             seasonality_mode="multiplicative",
#             changepoint_prior_scale=30,
#             seasonality_prior_scale=35,
#             holidays_prior_scale=20,
#             daily_seasonality=False,
#             weekly_seasonality=False,
#             yearly_seasonality=False,
#             ).add_seasonality(
#                 name='monthly',
#                 period=30.5,
#                 fourier_order=55
#             ).add_seasonality(
#                 name="daily",
#                 period=1,
#                 fourier_order=15
#             ).add_seasonality(
#                 name="weekly",
#                 period=7,
#                 fourier_order=20
#             ).add_seasonality(
#                 name="yearly",
#                 period=365.25,
#                 fourier_order=20
#             ).add_seasonality(
#                 name="quarterly",
#                 period = 365.25/4,
#                 fourier_order=5,
#                 prior_scale = 15)
#     else:
#         m = Prophet(growth = 'linear')

#     model = m.fit(company_prices[0:train_size])

#     future_dates = model.make_future_dataframe(periods = h)

#     if eliminate_weekends is not None:
#         future_dates['day'] = future_dates['ds'].dt.weekday
#         future_dates = future_dates[future_dates['day']<=4]
#     else:
#         pass

#     if save_model == True:
#         with open(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', 'w') as fout:
#             json.dump(model_to_json(model), fout)  # Save model

#     prediction = model.predict(future_dates)

#     return model, prediction, future_dates


#! function to retrain the model everyday at 15:30 with the new data

#! FUNCTION NOT USED
# def retrain_old_company_models(companies_present, holidays_list_path, h, eliminate_weekends):
#     company_prices, holidays_list = read_data(companies_present, 365*5, holidays_list_path) # read data for 5 years
#     # training_data = training_data[training_data['Company'] == companies_present]
#     train_size = len(company_prices)
#     train_val_size = train_size
#     model, prediction, future_dates = model_building_prophet(companies_present, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = True)
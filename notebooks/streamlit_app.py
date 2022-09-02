# make an automated model building pipeline that builds  model for a given company which is new in the dataset
# make list of comapnies using the companies given in training dataset

import json
import time
import os
import warnings
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
from functions import *
from datetime import datetime, timedelta, date

warnings.filterwarnings("ignore")
st.set_page_config(layout="wide")

st.sidebar.title('Daily Stock Price Prediction')

today = date.today()
yesterday = today - timedelta(days=1)

old_company_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')["Company"].unique()
with open("/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/config/process/companies_config.txt", "r") as f:
    new_companies_list=[i for line in f for i in line.split(',')]

new_company = list(set(new_companies_list) - set(old_company_list))
old_company = list(set(old_company_list) - set(new_companies_list))

if new_company:
    with st.spinner('Loading new stock...'):
        new_company = ','.join(new_company)
        eliminate_weekends = True
        h = 1
        company_prices, holidays_list = read_data(new_company, 365*5, '/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv') # read data for 5 years
        train_size = len(company_prices)
        train_val_size = train_size
        model_building_prophet(new_company, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = True)

if old_company:
    with st.spinner('Removing stock...'):
        old_company = ','.join(old_company)
        training_data = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')
        training_data = training_data[training_data['Company'] != old_company]
        training_data.to_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv', index=False)
        os.remove(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{old_company}.csv')
        os.remove(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{old_company}.json')

#! sidebar dropdown menu to select the company with cached data
old_company_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')["Company"].unique()
company = st.sidebar.selectbox('Select the company', old_company_list)

st.title('Live Stock Price Prediction')
with st.spinner('Making the prediction...'):
    Date = str(date.today())
    if os.path.exists(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv'):
        error_df = pd.read_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv')
    else:
        error_df = pd.DataFrame(columns=['Date', 'Actual_Close', 'Predicted_Close', 'Predicted_Close_Minimum', 'Predicted_Close_Maximum', 'Percent_Change_from_Close', 'Actual_Up_Down', 'Predicted_Up_Down', 'Company'])
        error_df = error_df.append({'Date': '07-04-2022'}, ignore_index=True)
        error_df.to_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv', index=False)
        error_df = pd.read_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv')

    if error_df.iloc[-1]['Date'] >= Date:
        error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', False)), next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', False), error_df, company)
    else:
        error_df = filling_missing_dates(error_df, company)
        error_df = pred_vs_real_comparision(real_stock_price(company, next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', False)), next_day_prediction(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', False), error_df, company)

# if today is a holiday remove today from error_df:

if is_holiday(today) == True:
    error_df = error_df[error_df['Date'] != Date]

#! saving the df to a csv file
error_df.to_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv', index=False)

#! filtering the dataframe as per the company selected
error_df = error_df[error_df['Company'] == company]

# error_df['Date'] = pd.to_datetime(error_df['Date'])
error_df = error_df.set_index('Date', drop = False)
error_df['Date'] = error_df.index
# error_df.index = error_df.index.date

#! making 3 columns
col1, col2, col3 = st.columns((1,1,1))

#! for col1 or yesterday's prediction
yesterday = datetime.now() - timedelta(days = 1)
yesterday1 = yesterday.date()
if yesterday.weekday() == 5 or yesterday.weekday() == 6 or is_holiday(yesterday1) == True:
    yesterday_date = date.today() - timedelta(days = 1)
    yesterday_date = str(yesterday_date.strftime('%d-%B-%Y'))
    col1.header('%s'%yesterday_date)
    col1.markdown('**Market Holiday**')

else:
    yesterday_date = date.today() - timedelta(days=1)
    yesterday_date = str(yesterday_date.strftime('%d-%B-%Y'))

    yesterday_close, yesterday_predicted_close, yesterday_actual_up_down, yesterday_predicted_up_down = print_metrics_columns(error_df, yesterday_date)

    col1.header('%s'%yesterday_date)
    col1.metric('Actual Price', '\u20B9' + str(yesterday_close), yesterday_actual_up_down)
    col1.metric('Predicted Price', '\u20B9' + str(yesterday_predicted_close), yesterday_predicted_up_down)

#! for col2 or today's prediction
todays_date = date.today()
todays_date = str(todays_date.strftime('%d-%B-%Y'))

if datetime.now().weekday() == 5 or datetime.now().weekday() == 6 or is_holiday(today) == True:
    col2.header('Today')
    col2.markdown('**Market Holiday**')

else:
    todays_close, todays_predicted_close, todays_actual_up_down, todays_predicted_up_down = print_metrics_columns(error_df, todays_date)

    col2.header('Today')
    if datetime.now().time() > datetime.strptime('15:31', '%H:%M').time():
        col2.metric('Actual Price', '\u20B9' + str(todays_close), todays_actual_up_down)
    col2.metric('Predicted Price', '\u20B9' + str(todays_predicted_close), todays_predicted_up_down)

#! for col3 or tomorrow's prediction
if datetime.today().weekday() not in [5, 6] or is_holiday(today) == False:
    tomorrows_date = date.today() + timedelta(days=1)
    tomorrows_date = str(tomorrows_date.strftime('%d-%B-%Y'))

    tomorrows_predicted_close = (error_df[error_df['Date'] == tomorrows_date]['Predicted_Close'].values[0]).round(2)
    tomorrows_predicted_up_down = error_df[error_df['Date'] == tomorrows_date]['Predicted_Up_Down'].values[0]
    if tomorrows_predicted_up_down == 'Up':
        tomorrows_predicted_up_down =  '+Up'
    elif tomorrows_predicted_up_down == 'Down':
        tomorrows_predicted_up_down = '-Down'

    col3.header('%s'%tomorrows_date)
    col3.metric('Predicted Price','\u20B9' + str(tomorrows_predicted_close), tomorrows_predicted_up_down)

else:
    if datetime.today().weekday() == 5:
        monday_date = date.today() + timedelta(days=2)
    elif datetime.today().weekday() != 5 or is_holiday == True:
        monday_date = date.today() + timedelta(days=1)

    monday_date = str(monday_date.strftime('%d-%B-%Y'))
    mondays_predicted_close = (error_df[error_df['Date'] == monday_date]['Predicted_Close'].values[0]).round(2)
    mondays_predicted_up_down = error_df[error_df['Date'] == monday_date]['Predicted_Up_Down'].values[0]
    if mondays_predicted_up_down == 'Up':
        tomorrows_predicted_up_down =  '+Up'
    elif mondays_predicted_up_down == 'Down':
        tomorrows_predicted_up_down = '-Down'

    col3.header('%s'%monday_date)
    col3.metric('Predicted Price','\u20B9' + str(mondays_predicted_close), tomorrows_predicted_up_down)

#! printing the next prediction day and time
if datetime.now().weekday() == 5:
    one_day_later_date = datetime.now() + timedelta(days=2)
elif datetime.now().weekday() == 6 or is_holiday(today) == True:
    one_day_later_date = datetime.now() + timedelta(days=1)
else:
    if datetime.now().hour < 15 :
        one_day_later_date = (datetime.now())
    else:
        one_day_later_date = (datetime.now()) + timedelta(1)

one_day_later_date = one_day_later_date.replace(hour = 15, minute = 31, second = 00, microsecond = 00)
one_day_later_date = one_day_later_date.strftime('%d-%B-%Y @ %H:%M:%S')

st.sidebar.header('Next Prediction:')
st.sidebar.success(one_day_later_date)

#! line chart for actual and predicted values
st.subheader('Actual vs Predicted Prices')
plot = px.line(error_df, x='Date', y=['Actual_Close', 'Predicted_Close'], title='Actual vs Predicted Prices', markers = True)
st.plotly_chart(plot, use_container_width=True)

#! writing the accuracy of the model
new_accuracy = compute_accuracy(error_df['Actual_Up_Down'].values, error_df['Predicted_Up_Down'].values)
st.sidebar.header('Accuracy of Predicting Up/Down:')
st.sidebar.success(f"{new_accuracy}%")
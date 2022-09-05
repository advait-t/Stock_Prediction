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
import warnings
warnings.filterwarnings("ignore")

def model_building_for_new_company(company, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = False):

    if holidays_list is not None:

        # variables for the model building and their meaning:
        '''
        holidays: list, list of holidays
        n_changepoints: int, number of changepoints. Change points are abrupt variations in time series data. (n_changepoints = 1 means there is only one changepoint.)
        n_changepoints_scale: float, scale of the number of changepoints 
        changepoint_prior_scale: float, scale of the changepoint prior
        yearly_seasonality: boolean, True if yearly seasonality is to be used, False otherwise
        weekly_seasonality: boolean, True if weekly seasonality is to be used, False otherwise
        daily_seasonality: boolean, True if daily seasonality is to be used, False otherwise
        holidays_prior_scale: float, scale of the holiday prior
        holidays_yearly_prior_scale: float, scale of the yearly holiday prior
        fourier_order: int, order of the fourier series. How quickly the seasonility of the time series can change.
        '''

        m = Prophet(growth="linear",
            holidays= holidays_list,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=30,
            seasonality_prior_scale=35,
            holidays_prior_scale=20,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=55
            ).add_seasonality(
                name="daily",
                period=1,
                fourier_order=15
            ).add_seasonality(
                name="weekly",
                period=7,
                fourier_order=20
            ).add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=20
            ).add_seasonality(
                name="quarterly",
                period = 365.25/4,
                fourier_order=5,
                prior_scale = 15)
    else:
        m = Prophet(growth = 'linear')

    model = m.fit(company_prices[0:train_size])

    future_dates = model.make_future_dataframe(periods = h)

    if eliminate_weekends is not None:
        future_dates['day'] = future_dates['ds'].dt.weekday
        future_dates = future_dates[future_dates['day']<=4]
    else:
        pass

    if save_model == True:
        with open(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/models/{company}.json', 'w') as fout:
            json.dump(model_to_json(model), fout)  # Save model

    prediction = model.predict(future_dates)
    return model, prediction, future_dates
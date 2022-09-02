import os
import warnings
import pandas as pd
# from functions import *
from datetime import date
import logging
from data_fetching import *
from model_building import *
from functions_for_inferencing import *

def main_func():

    today = date.today()

    old_company_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')["Company"].unique()
    with open("/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/config/process/companies_config.txt", "r") as f:
        new_companies_list=[i for line in f for i in line.split(',')]

    new_company = list(set(new_companies_list) - set(old_company_list))
    old_company = list(set(old_company_list) - set(new_companies_list))

    if new_company:
        #! reading data for new company
        company_prices = fetch_data_new_company(new_company)

        train_size = len(company_prices)
        train_val_size = train_size
        holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
        eliminate_weekends = True
        h = 1
        #! train for new company
        model_building_for_new_company(new_company, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = True)

    if old_company:
        #! delete data for old company
        data_delete_old_company(old_company)


    company_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/training_data.csv')["Company"].unique()

    for company in company_list:
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

        if is_holiday(today) == True:
            error_df = error_df[error_df['Date'] != Date]

        #! saving the df to a csv file
        error_df.to_csv(f'/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/error_df1{company}.csv', index=False)

    logging.info('Finished')


if __name__ == '__main__':
    main_func()
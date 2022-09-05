import os
import warnings
import pandas as pd
from datetime import date
from data_fetching import *
from model_building import *
from functions_for_inferencing import *
from inferencing import *
import warnings
warnings.filterwarnings("ignore")

def main():
    #! Checking if there is an addition or deletion of companies in the configs file
    new_company, delete_company = check_for_changes_in_companies()

    if new_company:
        #! reading data for new company
        company_prices = fetch_data_new_company(new_company)

        train_size = len(company_prices)
        train_val_size = train_size
        holidays_list = pd.read_csv('/Users/advait_t/Desktop/Jio/Stock_Prediction/Stock_Prediction/data/final/2017-2022_Holidays_NSE_BSE_EQ_EQD.csv')
        eliminate_weekends = True
        h = 1
        #! train model for new company
        model_building_for_new_company(new_company, company_prices, holidays_list, h, train_size, eliminate_weekends, save_model = True)

    if delete_company:
        #! delete data for old company
        data_delete_old_company(delete_company)

    #! inferencing the model
    inferencing()

if __name__ == "__main__":
    main()

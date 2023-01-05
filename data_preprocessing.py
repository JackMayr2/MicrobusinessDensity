import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

census_starter_df = pd.read_csv('godaddy-microbusiness-density-forecasting/census_starter.csv')
train_df = pd.read_csv('godaddy-microbusiness-density-forecasting/train.csv')

census_starter_df = census_starter_df.dropna()
train_df = train_df.dropna()
# Create a df accounting for the two-year lag in microdensity calculations.
# The base of lagged_df will be the train_csv
lagged_df = train_df.copy()

# We want to identify the year of the row
new = lagged_df["first_day_of_month"].str.split("-", n=2, expand=True)
lagged_df['Year'] = new[0].apply(float)
lagged_df['Month'] = new[1].apply(float)

# We want to match up cfips from train_df and census_starter_df and keep the columns that match up with the lag
# We will have the following columns added to lagged_df:
# pct_broadband, pct_college, pct_foreign_born, pct_it_workers, median_hh_income

lagged_df['pct_broadband'] = np.NaN
lagged_df['pct_college'] = np.NaN
lagged_df['pct_foreign_born'] = np.NaN
lagged_df['pct_it_workers'] = np.NaN
lagged_df['median_hh_income'] = np.NaN

lagged_df = lagged_df.reset_index()


# append the correct census lags to the density so we can create a DNN around them

def determine_lag(year, cfips, census_df):
    try:
        census_year = year - 2
        corresponding_columns = np.array(census_df.columns[census_df.columns.str.contains(str(census_year))]).tolist()
        year_values = census_df.loc[census_df['cfips'] == cfips, corresponding_columns]
        year_values = np.array(year_values).flatten()
        print(census_year, cfips, year_values[0], year_values[1], year_values[2], year_values[3], year_values[4])
        # return year_values[0], year_values[1], year_values[2], year_values[3], year_values[4]
        return year_values[0], year_values[1], year_values[2], year_values[3], year_values[4]
    except:
        print("no values")
        return np.NaN, np.NaN, np.NaN, np.NaN, np.NaN


for row in range(len(lagged_df.index)):
    lagged_df.loc[row, 'pct_broadband'], lagged_df.loc[row, 'pct_college'], lagged_df.loc[row, 'pct_foreign_born'], \
    lagged_df.loc[
        row, 'pct_it_workers'], lagged_df.loc[row, 'median_hh_income'] = zip(
        *[determine_lag(int(lagged_df.loc[row, 'Year']), int(lagged_df.loc[row, 'cfips']), census_starter_df)])

lagged_df.to_csv('Generated-Data/lagged_data.csv', index=False)

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

processed_data = 'Generated-Data/lagged_data.csv'
data_df = pd.read_csv(processed_data)

sns.pairplot(data_df[['microbusiness_density', 'pct_broadband', 'pct_college', 'pct_foreign_born', 'median_hh_income']], diag_kind='kde')
plt.show()

data_descrip = data_df.describe().transpose()


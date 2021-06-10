
import pandas as pd
import numpy as np
from pathlib import Path
import os

# set path
SRC = Path(os.getcwd()).resolve() / "src"
BLD = SRC.parent / "bld"

# policy stringency index
policy_data = pd.read_csv(SRC / "original_data"/ "OxCGRT_latest.csv").query("CountryCode=='NLD'")
policy_data['Date'] = pd.to_datetime(policy_data['Date'], format='%Y%m%d', errors='ignore')
policy_data = policy_data.set_index('Date')
policy_stringency = policy_data[['C1_School closing','C2_Workplace closing','C6_Stay at home requirements']]
# policy_stringency.plot(kind = 'line')


covid_data_wave2 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_04.pickle")
covid_data_wave3 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_05.pickle")
covid_data_wave5 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_09.pickle")
covid_data_wave6 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_12.pickle")
background = pd.read_pickle(SRC / "original_data" / "liss" / "background_data_merged.pickle")

covid_data = pd.concat([covid_data_wave2, covid_data_wave3, covid_data_wave5, covid_data_wave6])
select_background = background.loc[:, ["hh_id","age","female","education_cbs","net_income","hh_members",
                                       "hh_children"]].dropna(axis=0, how='any')
select_covid = covid_data.loc[:, ["infection_diagnosed"]]

merge_data = select_covid.join(select_background, on="personal_id", how="inner")
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) if "infection_diagnosed" in merge_data.columns else np.nan # 1: infection diagnosed, 0: not infected or not diagnosed

infected_percent = merge_data.groupby('month').sum()['infected'] / merge_data.groupby('month').count()['infected']

stringency_stat = pd.DataFrame({
    "April": policy_stringency.loc['2020-04-01':'2020-04-30',:].mean(),
"May": policy_stringency.loc['2020-05-01':'2020-05-31',:].mean(),
"September": policy_stringency.loc['2020-09-01':'2020-09-30',:].mean(),
"December": policy_stringency.loc['2020-12-01':'2020-12-31',:].mean()})

stringency_stat.to_csv(BLD / "tables" / "policy_stringency.csv")

np.corrcoef(stringency_stat.sum(), infected_percent)
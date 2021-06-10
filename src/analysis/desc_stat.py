
import statsmodels.api as sm
import numpy as np
import pandas as pd
from pathlib import Path
import os

# set path
SRC = Path(os.getcwd()).resolve() / "src"
BLD = SRC.parent / "bld"

# read liss data
background = pd.read_pickle(SRC / "original_data" / "liss" / "background_data_merged.pickle")
covid_data_wave1 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_03.pickle")
covid_data_wave2 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_04.pickle")
covid_data_wave3 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_05.pickle")
covid_data_wave5 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_09.pickle")
covid_data_wave6 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_12.pickle")
wave_month_dct = {1:'03',2:'04',3:'05',4:'06',5:'09',6:'12'}

# select variables from background and covid_data
select_background = background.loc[:, ["age","female","edu_4","net_income","hh_members",
                                       "hh_children"]].dropna(axis=0, how='any')

# infected


def get_desc_y(covid_data, month):
    infected = covid_data.loc[:, ["infection_diagnosed"]]
    infected['infected'] = infected["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(
        int)  # 1: infection diagnosed, 0: not infected or not diagnosed
    merge_data = infected.join(select_background, on="personal_id", how="inner")
    y_var = merge_data['infected']
    return pd.DataFrame({"Month": month, "Observations": y_var.shape[0], "Total infected": y_var.sum()}, index=[0])

desc_y_var = pd.concat([
    get_desc_y(covid_data_wave2, "April"),
    get_desc_y(covid_data_wave3, "May"),
    get_desc_y(covid_data_wave5, "September"),
    get_desc_y(covid_data_wave6, "December"),
]).reset_index(drop=True)
desc_y_var.to_csv(BLD / "tables" / "stat_infected_y_var.csv")


covid_data = pd.concat([covid_data_wave2, covid_data_wave3, covid_data_wave5, covid_data_wave6])
infected = covid_data.loc[:, ["infection_diagnosed"]]
infected['infected'] = infected["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) # 1: infection diagnosed, 0: not infected or not diagnosed
merge_data = infected.join(select_background, on="personal_id", how="inner")


# compliance
# compliance = covid_data_wave1[['avoid_busy_places','avoid_public_places','maintain_distance','adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms','no_avoidance_behaviors', 'comply_curfew_self']].reset_index(level='month',drop=True).dropna(axis=0, how='any')
# compliance.insert(7,"compliance_index", 0)
# compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"), 'compliance_index'] = compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"),
#                                                                                                              ['avoid_busy_places','avoid_public_places','maintain_distance',
#                                                                                                               'adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms']].sum(axis='columns')
# merge_data = select_background.join(compliance[["compliance_index"]], on="personal_id", how="inner")
# y_var = merge_data['compliance_index']
# desc_y_var = pd.DataFrame({"Month": "Marth", "Observations": y_var.shape[0], "Total": y_var.sum(),
#                            "Mean": y_var.mean(),
#                            "Std": y_var.std(),
#                            "Min": y_var.min(),
#                            "Max": y_var.max()
#                            },index=[0])
# desc_y_var.to_csv(BLD / "tables" / "stat_compliance_y_var.csv")
####################
#### data clean ####
####################
# cut age
merge_data["age_cut"] = pd.cut(merge_data.age, [0,25,50,75,100], right=False)

# cut net income
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 1000))+[5001, 7500, np.inf],
                                      right=False)

# refine codes of category

merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # not alone dummy 1: not alone, 0: alone
merge_data['living_with_children'] = (merge_data['hh_children']>0).astype(int)  # 1: has children, 0: no children
merge_data['age'] = merge_data['age'].astype('int')
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) if "infection_diagnosed" in merge_data.columns else np.nan # 1: infection diagnosed, 0: not infected or not diagnosed
merge_data['net_income'] = merge_data['net_income'].clip(1, 6000)
# anti dummy
merge_data['living_alone'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1 - merge_data['female']

####################
##  prepare x, y  ##
####################
# select x and y from merge_data
# y = merge_data.loc[:, "infected"]
# y = merge_data.loc[:, "compliance_index"]

x = merge_data.loc[:, ['female','living_alone']]
x['living_with_children'] = merge_data['living_with_children']


# change category to dummy and drop first dummy
x = x.join(pd.get_dummies(merge_data['edu_4'].cat.remove_unused_categories(), prefix='education', drop_first=False, prefix_sep=':'))
x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age', drop_first=False, prefix_sep=':'))

x['log_net_income'] = np.log(merge_data['net_income'])



####################
## Descriptive statistics ##
####################
# x_var = x.reset_index(level='personal_id').reset_index('month', drop=True).drop_duplicates(inplace=False).drop(columns='personal_id')
x_var = x.reset_index(level='personal_id').drop_duplicates(inplace=False).drop(columns='personal_id')
df_desc_stat = pd.DataFrame({
    "Observations": x_var.count(),
    "Mean": x_var.mean(),
    "Std": x_var.std(),
    "Min": x_var.min(),
    "Max": x_var.max(),
    "Sum": x_var.sum()
})
# df_desc_stat.to_csv(BLD / "tables" / "stat_infected_x_var.csv")
df_desc_stat.to_csv(BLD / "tables" / "stat_compliance_x_var.csv")

infected = pd.concat([covid_data_wave2, covid_data_wave3, covid_data_wave5, covid_data_wave6]).loc[:, ["infection_diagnosed"]].dropna()
compliance = covid_data_wave1[['avoid_busy_places','avoid_public_places','maintain_distance','adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms','no_avoidance_behaviors', 'comply_curfew_self']].reset_index(level='month',drop=True).dropna(axis=0, how='any')


infected_age = infected.join(select_background, on="personal_id", how="inner")['age'].reset_index(level='personal_id').drop_duplicates(inplace=False).set_index('personal_id')['age']

compliance_age = compliance.join(select_background, on="personal_id", how="inner")['age'].reset_index(level='personal_id').drop_duplicates(inplace=False).set_index('personal_id')['age']

pd.DataFrame({"Infected Age": infected_age,"Compliance Age": compliance_age}).boxplot()

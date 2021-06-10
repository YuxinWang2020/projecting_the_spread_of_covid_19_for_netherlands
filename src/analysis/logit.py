
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
covid_data_wave4 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_06.pickle")
covid_data_wave5 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_09.pickle")
covid_data_wave6 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_12.pickle")
wave_month_dct = {1:'03',2:'04',3:'05',4:'06',5:'09',6:'12'}

# concat wave3, wave5 and wave6 by row
# covid_data = pd.concat([covid_data_wave2, covid_data_wave3])
# covid_data = pd.concat([covid_data_wave5, covid_data_wave6])
covid_data = pd.concat([covid_data_wave2, covid_data_wave3, covid_data_wave5, covid_data_wave6])
# covid_data = covid_data_wave5

# select variables from background and covid_data
select_background = background.loc[:, ["age","female","education_cbs","net_income","hh_members",
                                       "hh_children","location_urban"]].dropna(axis=0, how='any')
# select_covid = covid_data.loc[:, ["infection_diagnosed", "work_status"]]
select_covid = covid_data.loc[:, ["infection_diagnosed"]]
# merge select_background and select_covid by column
merge_data = select_covid.join(select_background, on="personal_id", how="inner")
# merge_data = select_background

# merge essential_worker
# essential_worker = covid_data_wave2['essential_worker'].reset_index(level='month',drop=True)
# merge_data = merge_data.join(essential_worker, on="personal_id", how="inner").dropna(axis=0, how='any')

# compliance
# compliance = covid_data_wave1[['avoid_busy_places','avoid_public_places','maintain_distance','adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms','no_avoidance_behaviors', 'comply_curfew_self']].reset_index(level='month',drop=True).dropna(axis=0, how='any')
# compliance.insert(7,"compliance_index", 0)
# compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"), 'compliance_index'] = compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"),
#                                                                                                              ['avoid_busy_places','avoid_public_places','maintain_distance',
#                                                                                                               'adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms']].sum(axis='columns')
# compliance["compliance_dummy"] = (compliance["compliance_index"] >= 4).astype(int)
# merge_data = merge_data.join(compliance[["compliance_index","compliance_dummy"]], on="personal_id", how="inner")
####################
#### data clean ####
####################
# cut age by 10
# merge_data["age_group_by10"] = pd.cut(merge_data.age, range(0, 105, 10),
#                                       right=False,
#                                       labels=["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)])
merge_data["age_cut"] = pd.cut(merge_data.age, [0,25,50,75,100], right=False)

# cut net income by 500
# merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 1000))+[5001, 7500, np.inf],
#                                       right=False,
#                                       labels=["0"]+["{0} - {1}".format(i, i + 999) for i in list(range(1, 5000, 1000))]+["5001 - 7500",">7500"])
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 1000))+[5001, 7500, np.inf],
                                      right=False)

# refine codes of category
merge_data['edu_index'] = merge_data['education_cbs'].cat.codes.replace(
    {0:1, 1:0, 2:0, 3:0, 4:0, 5:2, 6:0, 7:0})  # 1:'primary school', 2:'wo (university)'
# merge_data['age_index'] = merge_data['age_group'].cat.codes.replace({0:1, 1:2, 2:3}) # 1: '<40', 2: '40 to 65', 3: '>65'
# merge_data['age_by10_index'] = merge_data['age_group_by10'].cat.codes.replace(
#     {i:(i+1) for i in dict(enumerate(merge_data['age_group_by10'].cat.categories)).keys()}) # 1: '0 - 9', 2: '10 - 19', ...
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # not alone dummy 1: not alone, 0: alone
# merge_data['dom_situation_dummy'] = merge_data['dom_situation'].cat.codes.replace({2:1,3:1,4:1}) # 1: co-habitation, 0: single
merge_data['hh_children_dummy'] = (merge_data['hh_children']>0).astype(int)  # 1: has children, 0: no children
# merge_data['location_urban_index'] = 5 - merge_data['location_urban'].cat.codes  # 1:'not urban' ~ 5: 'Extremely urban'
merge_data['age'] = merge_data['age'].astype('int')
# merge_data['working_dummy'] = merge_data['work_status'].isin(['employed', 'self-employed']).astype(int) # 1: employed, 0: others
# merge_data['school_dummy'] = merge_data['work_status'].eq('student or trainee').astype(int) # 1: student, 0: others
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) if "infection_diagnosed" in merge_data.columns else np.nan # 1: infection diagnosed, 0: not infected or not diagnosed
merge_data['net_income_index'] = merge_data['net_income_cat'].cat.codes
merge_data['net_income'] = merge_data['net_income'].clip(1, 6000)
# anti dummy
merge_data['alone_dummy'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1 - merge_data['female']
# merge_data['single_dummy'] = 1 - merge_data['dom_situation_dummy']
merge_data['member_not_child_dummy'] = (merge_data['hh_members_dummy'] & (merge_data['hh_children_dummy'] == 0)).astype(int)

####################
##  prepare x, y  ##
####################
# select x and y from merge_data
y = merge_data.loc[:, "infected"]
# y = merge_data.loc[:, "compliance_index"]

# x = merge_data.loc[:, ['age','female','hh_members_dummy','hh_children_dummy','working_dummy', 'school_dummy', 'net_income_index']]
x = merge_data.loc[:, ['female','alone_dummy']]
x['hh_children_dummy'] = merge_data['hh_children_dummy']


# x['age_mul_10'] = merge_data['age']/10
x = sm.add_constant(x) # add constant if need intercept

# change category to dummy and drop first dummy
x = x.join(pd.get_dummies(merge_data['education_cbs'].cat.remove_unused_categories(), prefix='education_cbs', drop_first=True))
# x = x.join(pd.get_dummies(merge_data['location_urban'], prefix='location_urban', drop_first=True))
x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age_cut', drop_first=True))
# x = x.join(pd.get_dummies(merge_data['net_income_cat'], prefix='net_income_cat', drop_first=True))

# add square of age
# x.insert(1, 'age_square', x.loc[:, 'age'].pow(2))
x['net_income_log'] = np.log(merge_data['net_income'])

# def checkVIF_new(df):
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#     df = df.copy()
#     name = df.columns
#     VIF_list = [variance_inflation_factor(df.values,i) for i in range(df.shape[1])]
#     VIF = pd.DataFrame({"VIF":VIF_list}, index=name)
#     return VIF
# print(checkVIF_new(x))

# add interaction
# x['university_age_25_50'] = x['age_cut_[25, 50)'] * x['education_cbs_wo (university)']
x['university_age_50_75'] = x['age_cut_[50, 75)'] * x['education_cbs_wo (university)']
# x['university_age_75_100'] = x['age_cut_[75, 100)'] * x['education_cbs_wo (university)']
x['alone_dummy_age_25_50'] = x['age_cut_[25, 50)'] * x['alone_dummy']
# x['alone_dummy_age_50_75'] = x['age_cut_[50, 75)'] * x['alone_dummy']
# x['alone_dummy_age_75_100'] = x['age_cut_[75, 100)'] * x['alone_dummy']
# x['hh_children_dummy_age_25_50'] = x['age_cut_[25, 50)'] * x['hh_children_dummy']
# x['hh_children_dummy_age_50_75'] = x['age_cut_[50, 75)'] * x['hh_children_dummy']
# x['hh_children_dummy_age_75_100'] = x['age_cut_[75, 100)'] * x['hh_children_dummy']
# x['female_age_50_75'] = x['age_cut_[50, 75)'] * x['female']
# x['edu_female'] = x['female'] * x['education_cbs_wo (university)']
# x['edu_female_age_50_75'] = x['age_cut_[50, 75)'] * x['female'] * x['education_cbs_wo (university)']

# x['very_urban_age_50_75'] = x['age_cut_[50, 75)'] * x['location_urban_Very urban']
# x['income_age_50_75'] = x['age_cut_[50, 75)'] * x['net_income']
x['income_log_age_50_75'] = x['age_cut_[50, 75)'] * x['net_income_log']
# x['complaince_age_50_75'] = x['age_cut_[50, 75)'] * x['compliance_dummy']


####################
## Descriptive statistics ##
####################
# df_desc_stat = pd.DataFrame()
# for wave,month in wave_month_dct.items():
#     covid_data = globals()['covid_data_wave'+str(wave)]
#     df_desc_stat = pd.concat([df_desc_stat, pd.Series({
#         'Month':month,
#         'Observations':covid_data.shape[0],
#         'total_infected':covid_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').sum() if
#     })])
xy = pd.DataFrame(y).join(x.drop(columns='const'))
df_desc_stat = pd.DataFrame({
    "Observations": xy.count(),
    "Mean": xy.mean(),
    "Std": xy.std(),
    "Min": xy.min(),
    "Max": xy.max(),
    "Sum": xy.sum()
})
df_desc_stat.to_csv(BLD / "tables" / "desc_stat_infected.csv")
####################
## run regression ##
####################
model = sm.Logit(y, x)
# model = sm.OLS(y, x) # logit model
result = model.fit()
print(result.summary())
pd.DataFrame(result.summary().tables[1]).to_csv(BLD / "tables" / "infected_logit.csv")
# pd.DataFrame(result.summary().tables[1]).to_csv(BLD / "tables" / "compliance_ols.csv")

# get Odds Ratio
conf = result.conf_int()
conf['Odds Ratio'] = result.params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))
np.exp(conf).to_csv(BLD / "tables" / "infected_logit_OR.csv")

# policy stringency index
# policy_data = pd.read_csv(SRC / "original_data"/ "OxCGRT_latest.csv").query("CountryCode=='NLD'")
# policy_data['Date'] = pd.to_datetime(policy_data['Date'], format='%Y%m%d', errors='ignore')
# policy_data = policy_data.set_index('Date')
# policy_stringency = policy_data[['C1_School closing','C2_Workplace closing','C6_Stay at home requirements']]
# policy_stringency.plot(kind = 'line')

# save x and y to dta
# x.join(y).to_stata(BLD / 'data' / 'liss' / 'logit_merge_data.dta')
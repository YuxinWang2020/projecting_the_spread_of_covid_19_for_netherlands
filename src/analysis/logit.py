
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

# concat wave3, wave5 and wave6 by row
covid_data = pd.concat([covid_data_wave3, covid_data_wave5, covid_data_wave6])

# select variables from background and covid_data
select_background = background.loc[:, ["hh_id","age","age_group","female","education_cbs","net_income","hh_members",
                                       "hh_children","dom_situation","location_urban"]].dropna(axis=0, how='any')
select_covid = covid_data.loc[:, ["infection_diagnosed", "work_status"]]
# merge select_background and select_covid by column
merge_data = select_covid.join(select_background, on="personal_id", how="inner")

####################
#### data clean ####
####################
# cut age by 10
merge_data["age_group_by10"] = pd.cut(merge_data.age, range(0, 105, 10),
                                      right=False,
                                      labels=["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)])
# cut net income by 500
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 500))+[5001, 7500, np.inf],
                                      right=False,
                                      labels=["0"]+["{0} - {1}".format(i, i + 499) for i in list(range(1, 5000, 500))]+["5001 - 7500",">7500"])

# refine codes of category
merge_data['edu_index'] = merge_data['education_cbs'].cat.codes.replace(
    {0:1, 1:0, 2:0, 3:0, 4:0, 5:2, 6:0, 7:0})  # 1:'primary school', 2:'wo (university)'
merge_data['age_index'] = merge_data['age_group'].cat.codes.replace({0:1, 1:2, 2:3}) # 1: '<40', 2: '40 to 65', 3: '>65'
merge_data['age_by10_index'] = merge_data['age_group_by10'].cat.codes.replace(
    {i:(i+1) for i in dict(enumerate(merge_data['age_group_by10'].cat.categories)).keys()}) # 1: '0 - 9', 2: '10 - 19', ...
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # not alone dummy 1: not alone, 0: alone
merge_data['dom_situation_dummy'] = merge_data['dom_situation'].cat.codes.replace({2:1,3:1,4:1}) # 1: co-habitation, 0: single
merge_data['hh_children_dummy'] = (merge_data['hh_children']>0).astype(int)  # 1: has children, 0: no children
merge_data['location_urban_index'] = 5 - merge_data['location_urban'].cat.codes  # 1:'not urban' ~ 5: 'Extremely urban'
merge_data['age'] = merge_data['age'].astype('int')
merge_data['working_dummy'] = merge_data['work_status'].isin(['employed', 'self-employed']).astype(int) # 1: employed, 0: others
merge_data['school_dummy'] = merge_data['work_status'].eq('student or trainee').astype(int) # 1: student, 0: others
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) # 1: infection diagnosed, 0: not infected or not diagnosed
merge_data['net_income_index'] = merge_data['net_income_cat'].cat.codes
# anti dummy
merge_data['alone_dummy'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1 - merge_data['female']
merge_data['single_dummy'] = 1 - merge_data['dom_situation_dummy']

####################
##  prepare x, y  ##
####################
# select x and y from merge_data
y = merge_data.loc[:, "infected"]
x = merge_data.loc[:, ['age','female','hh_members_dummy','hh_children_dummy','working_dummy', 'school_dummy', 'net_income_index']]
x = sm.add_constant(x) # add constant if need intercept

# change category to dummy and drop first dummy
df_dummy = pd.get_dummies(merge_data['education_cbs'].cat.remove_unused_categories(), prefix='education_cbs', drop_first=True).join(
    pd.get_dummies(merge_data['location_urban'], prefix='location_urban', drop_first=True)
)
# add these dummies to x
x = x.join(df_dummy)
# add square of age
x.insert(1, 'age_square', merge_data.loc[:, 'age'].pow(2))
# save x and y to dta
x.join(y).to_stata(BLD / 'data' / 'liss' / 'logit_merge_data.dta')

####################
## run regression ##
####################
model = sm.Logit(y, x) # logit model
result = model.fit()
print(result.summary())

# get Odds Ratio
conf = result.conf_int()
conf['Odds Ratio'] = result.params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

import statsmodels.api as sm
import numpy as np
import pandas as pd
from pathlib import Path
import os

SRC = Path(os.getcwd()).resolve() / "src"
BLD = SRC.parent / "bld"

background = pd.read_pickle(SRC / "original_data" / "liss" / "background_data_merged.pickle")

# waves = ['02', '03', '04', '05', '06', '09', '12']
# for wave in waves:
#     covid_data = pd.read_pickle(SRC / "original_data" / "liss" / f"covid_data_2020_{wave}.pickle")
#     print([s for s in covid_data.columns.to_list() if (s.find("work_status")!=-1)])

# covid_data_wave2 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_04.pickle")
covid_data_wave3 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_05.pickle")
# covid_data_wave4 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_06.pickle")
covid_data_wave5 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_09.pickle")
covid_data_wave6 = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_12.pickle")

covid_data = pd.concat([covid_data_wave3, covid_data_wave5, covid_data_wave6])

# policy_stringency = pd.read_csv(SRC / "original_data" / "OxCGRT_latest.csv")

# select variables from background and covid_data
select_background = background.loc[:, ["hh_id","age","age_group","female","education_cbs","net_income","hh_members",
                                       "hh_children","dom_situation","location_urban"]].dropna(axis=0, how='any')
select_covid = covid_data.loc[:, ["infection_diagnosed", "work_status"]]
merge_data = select_covid.join(select_background, on="personal_id", how="inner")

labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]

merge_data["age_group_by10"] = pd.cut(merge_data.age, range(0, 105, 10), right=False, labels=labels)
# merge_data["age_older_70"] = pd.cut(merge_data.age, [0, 70, 105], right=False, labels=["<=70", ">70"])
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 500))+[5001, 7500, np.inf],
                                      right=False,
                                      labels=["0"]+["{0} - {1}".format(i, i + 499) for i in list(range(1, 5000, 500))]+["5001 - 7500",">7500"])


# refine codes of category
merge_data['edu_index'] = merge_data['education_cbs'].cat.codes.replace({0:1, 1:0, 2:0, 3:0, 4:0, 5:2, 6:0, 7:0})  # 1:'primary school', 2:'wo (university)'
merge_data['age_index'] = merge_data['age_group'].cat.codes.replace({0:1, 1:2, 2:3})
merge_data['age_by10_index'] = merge_data['age_group_by10'].cat.codes.replace({i:(i+1) for i in dict(enumerate(merge_data['age_group_by10'].cat.categories)).keys()})
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # 1: has other household member
merge_data['hh_children_dummy'] = (merge_data['hh_children']>0).astype(int)  # 1: has children
merge_data['location_urban_index'] = 5 - merge_data['location_urban'].cat.codes  # 1:'not urban' ~ 5: 'Extremely urban'
merge_data['dom_situation_dummy'] = merge_data['dom_situation'].cat.codes.replace({2:1,3:1,4:1})
merge_data['age'] = merge_data['age'].astype('int')
merge_data['working_dummy'] = merge_data['work_status'].isin(['employed', 'self-employed']).astype(int)
merge_data['school_dummy'] = merge_data['work_status'].eq('student or trainee').astype(int)
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int)
merge_data['net_income_index'] = merge_data['net_income_cat'].cat.codes

merge_data['alone_dummy'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1- merge_data['female']
merge_data['dom_not_single'] = 1- merge_data['dom_situation_dummy']

y = merge_data.loc[:, "infected"]
x = merge_data.loc[:, ['age','female','hh_members_dummy','hh_children_dummy','working_dummy', 'school_dummy', 'net_income_index']]
x = sm.add_constant(x) # add constant if need intercept

df_dummy = pd.get_dummies(merge_data['education_cbs'].cat.remove_unused_categories(), prefix='education_cbs').join(
    pd.get_dummies(merge_data['location_urban'], prefix='location_urban')
)
df_dummy = df_dummy.drop(labels=['education_cbs_primary school','location_urban_Not urban'], axis=1)
x = x.join(df_dummy)

x.insert(1, 'age_square', merge_data.loc[:, 'age'].pow(2))
# x.insert(2,'age_index_cube', merge_data.loc[:, 'age_index'].pow(3))

x.join(y).to_stata(BLD / 'data' / 'liss' / 'logit_merge_data.dta')

model = sm.Logit(y, x)
result = model.fit()
print(result.summary())

# Odds Ratio
np.exp(result.params)
params = result.params
conf = result.conf_int()
conf['Odds Ratio'] = params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))
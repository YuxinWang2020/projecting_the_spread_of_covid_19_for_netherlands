import pandas as pd
import numpy as np
from pathlib import Path
import os

# set path
SRC = Path(os.getcwd()).resolve() / "src"
BLD = SRC.parent / "bld"

data = pd.read_pickle(SRC / "original_data" / "liss" / "covid_data_2020_03.pickle")

Q3_Q10 = data[['avoid_busy_places','avoid_public_places','maintain_distance','adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms','no_avoidance_behaviors', 'comply_curfew_self']].dropna(axis=0, how='any')
# Q3_Q10 = Q3_Q10.query('comply_curfew_self=="yes"')

Q3_Q10.insert(7,"sum", 0)
Q3_Q10.loc[Q3_Q10['no_avoidance_behaviors']!=1 & (Q3_Q10['comply_curfew_self'] != "no"), 'sum'] = Q3_Q10.loc[Q3_Q10['no_avoidance_behaviors']!=1 & (Q3_Q10['comply_curfew_self'] != "no"),
                                                                                                             ['avoid_busy_places','avoid_public_places','maintain_distance',
                                                                                                              'adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms']].sum(axis='columns')

background = pd.read_pickle(SRC / "original_data" / "liss" / "background_data_merged.pickle")
select_background = background.loc[:, ["hh_id","age","age_group","gender","female","education_cbs","hh_members","hh_children","dom_situation","location_urban"]].dropna(axis=0, how='any')
merge_data = Q3_Q10.join(select_background, on="personal_id", how="inner")

labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 100, 10)]
merge_data["age_group_by10"] = pd.cut(merge_data.age, range(0, 105, 10), right=False, labels=labels)
merge_data["age_older_70"] = pd.cut(merge_data.age, [0, 70, 105], right=False, labels=["<=70", ">70"])
merge_data[["age","age_group_by10","age_older_70"]]

merge_data['edu_index'] = merge_data['education_cbs'].cat.codes.replace({7:1,0:2,1:3,2:4,3:5,4:6,5:7,6:np.nan})
merge_data['age_index'] = merge_data['age_group'].cat.codes.replace({0:1, 1:2, 2:3})
merge_data['age_by10_index'] = merge_data['age_group_by10'].cat.codes.replace({i:(i+1) for i in dict(enumerate(merge_data['age_group_by10'].cat.categories)).keys()})
merge_data['dom_situation_dummy'] = merge_data['dom_situation'].cat.codes.replace({2:1,3:1,4:1})
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)
merge_data['hh_children_dummy'] = (merge_data['hh_children']>0).astype(int)
merge_data['location_urban_index'] = 5 - merge_data['location_urban'].cat.codes
merge_data['age'] = merge_data['age'].astype('int')
merge_data['male'] = 1- merge_data['female']
merge_data['dom_not_single'] = 1- merge_data['dom_situation_dummy']
# merge_data=merge_data.rename(columns={"hh_members":"Number of Household Members", "dom_situation_dummy":"Domestic Situation", "edu_index":"Education Level", "hh_children_dummy":"Has Children","location_urban_index":"Location Urban"})
merge_data.head(5)

older_70 = merge_data[merge_data["age_older_70"]==">70"]

import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 100


y = merge_data.loc[:, "sum"]
x = merge_data.loc[:, ['age','edu_index','female','dom_situation_dummy','hh_members_dummy','hh_children_dummy','location_urban_index']]
x.insert(1,'age_square', merge_data.loc[:,'age'].pow(2))
# x.insert(2,'age_cube', merge_data.loc[:, 'age'].pow(3))

x = sm.add_constant(x) # add constant if need intercept
model = sm.OLS(y, x).fit() # ols fit
print(model.summary()) # print fit result
# with open("../bld/tables/age_square.csv", 'w') as f:
#     f.write(model.summary().tables[1].as_csv())

import seaborn as sns
coef = model.params
sns.regplot(x="age", y="sum", data=merge_data, x_jitter = 0.2, y_jitter = 0.5, scatter_kws = {'alpha' : 1/3}, fit_reg=False)
x=np.arange(0,100, 0.1)
y=[coef['age_square']*i**2 + coef['age']*i + coef['const'] for i in x]
sns.lineplot(x, y)
plt.show()
# plt.savefig("../bld/figures/age_square.png")




y = merge_data.loc[:, "sum"]
x = merge_data.loc[:, ['age','edu_index','female','dom_situation_dummy','hh_members_dummy','hh_children_dummy','location_urban_index']]
x.insert(1,'age_square', merge_data.loc[:,'age'].pow(2))
# x.insert(2,'age_cube', merge_data.loc[:, 'age'].pow(3))

x = sm.add_constant(x) # add constant if need intercept
model = sm.OLS(y, x).fit() # ols fit
print(model.summary()) # print fit result
# with open("../bld/tables/age_square.csv", 'w') as f:
#     f.write(model.summary().tables[1].as_csv())

import seaborn as sns
coef = model.params
sns.regplot(x="age", y="sum", data=merge_data, x_jitter = 0.2, y_jitter = 0.5, scatter_kws = {'alpha' : 1/3}, fit_reg=False)
x=np.arange(0,100, 0.1)
y=[coef['age_square']*i**2 + coef['age']*i + coef['const'] for i in x]
sns.lineplot(x, y)
plt.show()
# plt.savefig("../bld/figures/age_square.png")





y = merge_data.loc[:, "sum"]
x = merge_data.loc[:, ['age_index','edu_index','female','dom_situation_dummy','hh_members_dummy','hh_children_dummy','location_urban_index']]
x.insert(1,'age_index_square', merge_data.loc[:,'age_index'].pow(2))
x.insert(2,'age_index_cube', merge_data.loc[:, 'age_index'].pow(3))

x = sm.add_constant(x) # add constant if need intercept
model = sm.OLS(y, x).fit() # ols fit
print(model.summary()) # print fit result
# with open("../bld/tables/age_index_cube.csv", 'w') as f:
#     f.write(model.summary().tables[1].as_csv())

import seaborn as sns
coef = model.params
sns.regplot(x="age_index", y="sum", data=merge_data, x_jitter = 0.2, y_jitter = 0.5, scatter_kws = {'alpha' : 1/3}, fit_reg=False)
x=np.arange(0.5,3.5, 0.1)
y=[coef['age_index_cube']*i**3 + coef['age_index_square']*i**2 + coef['age_index']*i + coef['const'] for i in x]
sns.lineplot(x, y)
plt.show()
# plt.savefig("../bld/figures/age_index_cube.png")





y = merge_data.loc[:, "sum"]
x = merge_data.loc[:, ['age_by10_index','edu_index','male','female','dom_situation_dummy','hh_members_dummy','hh_children_dummy','location_urban_index']]
x.insert(1,'age_by10_index_square', merge_data.loc[:,'age_by10_index'].pow(2))
x.insert(2,'age_by10_index_cube', merge_data.loc[:, 'age_by10_index'].pow(3))

x = sm.add_constant(x) # add constant if need intercept
model = sm.OLS(y, x).fit() # ols fit
print(model.summary()) # print fit result
# with open("../bld/tables/age_by10_index_cube.csv", 'w') as f:
#     f.write(model.summary().tables[1].as_csv())

import seaborn as sns

coef = model.params
sns.regplot(x="age_by10_index", y="sum", data=merge_data, x_jitter = 0.2, y_jitter = 0.5, scatter_kws = {'alpha' : 1/3}, fit_reg=False)
x=np.arange(1,10, 0.1)
y=[coef['age_by10_index_cube']*i**3 + coef['age_by10_index_square']*i**2 + coef['age_by10_index']*i + coef['const'] for i in x]
sns.lineplot(x, y)
plt.show()
# plt.savefig("../bld/figures/age_by10_index_cube.png")




y = older_70.loc[:, "sum"]
x = older_70.loc[:, ['edu_index','male','female','dom_situation_dummy','dom_not_single','hh_members_dummy','hh_children_dummy','location_urban_index']]

x = sm.add_constant(x) # add constant if need intercept
model = sm.OLS(y, x).fit() # ols fit
print(model.summary()) # print fit result
# with open("../bld/tables/age_older_70.csv", 'w') as f:
#     f.write(model.summary().tables[1].as_csv())












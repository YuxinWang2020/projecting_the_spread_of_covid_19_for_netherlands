
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
wave_month_dct = {1:'03',2:'04',3:'05',4:'06',5:'09',6:'12'}


# select variables from background and covid_data
select_background = background.loc[:, ["age","female","education_cbs","net_income","hh_members",
                                       "hh_children"]].dropna(axis=0, how='any')

# compliance
compliance = covid_data_wave1[['avoid_busy_places','avoid_public_places','maintain_distance','adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms','no_avoidance_behaviors', 'comply_curfew_self']].reset_index(level='month',drop=True).dropna(axis=0, how='any')
compliance.insert(7,"compliance_index", 0)
compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"), 'compliance_index'] = compliance.loc[compliance['no_avoidance_behaviors']!=1 & (compliance['comply_curfew_self'] != "no"),
                                                                                                             ['avoid_busy_places','avoid_public_places','maintain_distance',
                                                                                                              'adjust_school_work','quarantine_symptoms', 'quarantine_no_symptoms']].sum(axis='columns')
compliance["compliance_dummy"] = (compliance["compliance_index"] >= 4).astype(int)
merge_data = select_background.join(compliance[["compliance_index","compliance_dummy"]], on="personal_id", how="inner")
####################
#### data clean ####
####################
# cut age
merge_data["age_cut"] = pd.cut(merge_data.age, [0,25,50,75,100], right=False)

# cut net income
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 1000))+[5001, 7500, np.inf],
                                      right=False)

# refine codes of category
merge_data['edu_index'] = merge_data['education_cbs'].cat.codes.replace(
    {0:1, 1:0, 2:0, 3:0, 4:0, 5:2, 6:0, 7:0})  # 1:'primary school', 2:'wo (university)'
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # not alone dummy 1: not alone, 0: alone
merge_data['living_with_children'] = (merge_data['hh_children']>0).astype(int)  # 1: has children, 0: no children
merge_data['age'] = merge_data['age'].astype('int')
merge_data['net_income_index'] = merge_data['net_income_cat'].cat.codes
merge_data['net_income'] = merge_data['net_income'].clip(1, 6000)
# anti dummy
merge_data['living_alone'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1 - merge_data['female']
merge_data['member_not_child_dummy'] = (merge_data['hh_members_dummy'] & (merge_data['living_with_children'] == 0)).astype(int)

####################
##  prepare x, y  ##
####################
# select x and y from merge_data
y = merge_data.loc[:, "compliance_index"]

x = merge_data.loc[:, ['female','living_alone']]
x['living_with_children'] = merge_data['living_with_children']

x = sm.add_constant(x) # add constant if need intercept

# change category to dummy and drop first dummy
x = x.join(pd.get_dummies(merge_data['education_cbs'].cat.remove_unused_categories(), prefix='education', drop_first=True, prefix_sep=':'))
x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age', drop_first=True, prefix_sep=':'))

# x['net_income_log'] = np.log(merge_data['net_income'])

# add interaction
x['education:wo (university) # age:[50, 75)'] = x['age:[50, 75)'] * x['education:wo (university)']
x['living_alone # age:[75, 100)'] = x['age:[75, 100)'] * x['living_alone']

# x['log_net_income # age:[50, 75)'] = x['age:[50, 75)'] * x['log_net_income']


####################
## run regression ##
####################
model = sm.OLS(y, x)
result = model.fit()
print(result.summary())
summary = result.summary()

def summary_to_stata_format(summary):
    summary = pd.DataFrame(summary.tables[1].data)
    summary = summary.set_axis(summary.iloc[0].astype(str), axis='columns').set_index('').iloc[2:, [0, 1, 3]]
    for i in range(len(summary)):
        if float(summary.iloc[i, 2]) < 0.01:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '***'
        elif float(summary.iloc[i, 2]) < 0.05:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '**'
        elif float(summary.iloc[i, 2]) < 0.1:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '*'
        else:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip()
        summary.iloc[i, 1] = '(' + summary.iloc[i, 1].lstrip() + ')'
    summary = summary.drop(columns='P>|t|').stack()
    summary = summary.reset_index(level=1,drop=True)
    return summary

stata_format = summary_to_stata_format(summary)

infected_logit = pd.read_csv(BLD / "tables" / "compliance_ols.csv", index_col=0) if (BLD / "tables" / "compliance_ols.csv").exists() else pd.DataFrame()
infected_logit["March"] = stata_format

infected_logit.to_csv(BLD / "tables" / "compliance_ols.csv")


merge_data[['compliance_index','age_cut']].boxplot(column='compliance_index', by='age_cut')
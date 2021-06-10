
import statsmodels.api as sm
import numpy as np
import pandas as pd
from pathlib import Path
import os

# set path
SRC = Path(os.getcwd()).resolve() / "src"
BLD = SRC.parent / "bld"

wave = ["Pooled", "April & May", "September & December"][0]

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
if wave == "April & May":
    covid_data = pd.concat([covid_data_wave2, covid_data_wave3])
if wave == "September & December":
    covid_data = pd.concat([covid_data_wave5, covid_data_wave6])
if wave == "Pooled":
    covid_data = pd.concat([covid_data_wave2, covid_data_wave3, covid_data_wave5, covid_data_wave6])

# select variables from background and covid_data
select_background = background.loc[:, ["hh_id","age","female","edu_4","edu","net_income","hh_members",
                                       "hh_children","location_urban"]].dropna(axis=0, how='all')
select_covid = covid_data.loc[:, ["infection_diagnosed"]]
# merge select_background and select_covid by column
merge_data = select_covid.join(select_background, on="personal_id", how="inner")

####################
#### data clean ####
####################
# cut age
merge_data["age_cut"] = pd.cut(merge_data.age, [15,25,50,75,100], right=False)

# cut net income
merge_data["net_income_cat"] = pd.cut(merge_data['net_income'], [0]+list(range(1, 5000, 1000))+[5001, 7500, np.inf],
                                      right=False)

# refine codes of category
merge_data['hh_members_dummy'] = (merge_data['hh_members']>1).astype(int)  # not alone dummy 1: not alone, 0: alone
merge_data['living_with_children'] = (merge_data['hh_children']>0).astype(int)  # 1: has children, 0: no children
merge_data['age'] = merge_data['age'].astype('int')
merge_data['infected'] = merge_data["infection_diagnosed"].eq('yes, I have been diagnosed with it').astype(int) if "infection_diagnosed" in merge_data.columns else np.nan # 1: infection diagnosed, 0: not infected or not diagnosed
merge_data['net_income_index'] = merge_data['net_income_cat'].cat.codes
merge_data['net_income'] = merge_data['net_income'].clip(1, 6000)
merge_data['log_net_income'] = np.log(merge_data['net_income'])
# anti dummy
merge_data['living_alone'] = 1 - merge_data['hh_members_dummy']
merge_data['male'] = 1 - merge_data['female']
merge_data['member_not_child_dummy'] = (merge_data['hh_members_dummy'] & (merge_data['living_with_children'] == 0)).astype(int)

####################
##  prepare x, y  ##
####################
# select x and y from merge_data
y = merge_data.loc[:, "infected"]

x = merge_data.loc[:, ['female','living_alone','living_with_children','log_net_income']]

x = sm.add_constant(x) # add constant if need intercept

# change category to dummy and drop first dummy
x = x.join(pd.get_dummies(merge_data['edu_4'].cat.remove_unused_categories(), prefix='edu', drop_first=True, prefix_sep=':'))
x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age', drop_first=True, prefix_sep=':'))


# add interaction
# x['edu:tertiary # age:[50, 75)'] = x['age:[50, 75)'] * x['edu:tertiary']
x['living_alone # age:[25, 50)'] = x['age:[25, 50)'] * x['living_alone']

####################
## run regression ##
####################
model = sm.Logit(y, x)
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
    summary = summary.drop(columns='P>|z|').stack()
    summary = summary.reset_index(level=1,drop=True)
    return summary

stata_format = summary_to_stata_format(summary)

infected_logit = pd.read_csv(BLD / "tables" / "infected_logit.csv", index_col=0) if (BLD / "tables" / "infected_logit.csv").exists() else pd.DataFrame()
infected_logit[wave] = stata_format

infected_logit.to_csv(BLD / "tables" / "infected_logit.csv")

# get Odds Ratio
conf = result.conf_int()
conf['Odds Ratio'] = result.params
conf.columns = ['5%', '95%', 'Odds Ratio']
print(np.exp(conf))

infected_logit_OR = pd.read_csv(BLD / "tables" / "infected_logit_OR.csv", index_col=0) if (BLD / "tables" / "infected_logit_OR.csv").exists() else pd.DataFrame()
infected_logit_OR[wave] = np.exp(conf['Odds Ratio'])

infected_logit_OR.to_csv(BLD / "tables" / "infected_logit_OR.csv")


# bar plot
merge_data.query("infected == 1").age_cut.value_counts().sort_index().plot.bar(rot=0)

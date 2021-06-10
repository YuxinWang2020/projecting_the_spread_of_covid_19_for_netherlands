import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(SRC / "original_data" / "liss" / "background_data_merged.pickle")
@pytask.mark.produces(BLD / "data" / "liss" / "background.pickle")
def task_clean_background_data(depends_on, produces):

    background = pd.read_pickle(depends_on)
    select_background = background[["hh_id", "age", "female", "edu", "net_income", "hh_members",
                                    "hh_children", "location_urban"]].dropna(axis=0, how='any')
    select_background["age_cut"] = pd.cut(select_background.age,
                                          [15, 25, 50, 75, 120],
                                          right=False
                                          )
    select_background["net_income_cat"] = pd.cut(select_background['net_income'],
                                                 [0] + list(range(1, 5000, 1000)) + [5001, 7500, np.inf],
                                                 right=False)
    select_background['living_alone'] = (select_background['hh_members'] == 1).astype(
        int)  # alone dummy 1: alone, 0: not alone
    select_background['living_with_children'] = (select_background['hh_children'] > 0).astype(
        int)  # 1: has children, 0: no children
    select_background['age'] = select_background['age'].astype('int')
    select_background['hh_id'] = select_background['hh_id'].astype('int')
    select_background['female'] = select_background['female'].astype('int')
    select_background['net_income_index'] = select_background['net_income_cat'].cat.codes
    select_background['net_income'] = select_background['net_income'].clip(0, 6000)
    select_background['log_net_income'] = np.log(select_background['net_income'].clip(0.001, 6000))
    select_background['male'] = 1 - select_background['female']

    select_background.to_pickle(produces)


@pytask.mark.depends_on(
    {
        month: SRC / "original_data" / "liss" / f"covid_data_{month}.pickle"
        for month in ["2020_04", "2020_05", "2020_09", "2020_12"]
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "infected.pickle")
def task_clean_infected_data(depends_on, produces):

    covid_data_list = [pd.read_pickle(wave) for wave in depends_on.values()]
    covid_data = pd.concat(covid_data_list)
    select_covid = covid_data[["infection_diagnosed","infection_perceived"]].dropna(axis=0, how='all')
    select_covid['infection_diagnosed'] = select_covid['infection_diagnosed'].cat.rename_categories({'yes, I have been diagnosed with it':'yes'})
    # 1: infection diagnosed, 0: not infected diagnosed or unsure
    select_covid['infected'] = select_covid["infection_diagnosed"].eq('yes').astype(int)

    select_covid['Month'] = select_covid.index.get_level_values('month').month_name()
    select_covid.to_pickle(produces)


@pytask.mark.depends_on(
    {
        month: SRC / "original_data" / "liss" / f"covid_data_{month}.pickle"
        for month in ["2020_03"]
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "compliance.pickle")
def task_clean_compliance_data(depends_on, produces):

    covid_data_list = [pd.read_pickle(wave) for wave in depends_on.values()]
    covid_data = pd.concat(covid_data_list)
    compliance = covid_data[[
        'avoid_busy_places', 'avoid_public_places', 'maintain_distance', 'adjust_school_work', 'quarantine_symptoms',
        'quarantine_no_symptoms', 'no_avoidance_behaviors', 'comply_curfew_self'
    ]].dropna(axis=0, how='any')
    # compliance_index is sum of these compliance variables
    compliance['compliance_index'] = 0
    loc = (compliance['no_avoidance_behaviors'] != 1) & (compliance['comply_curfew_self'] != "no")
    compliance.loc[loc, 'compliance_index'] = compliance.loc[
        loc,
        ['avoid_busy_places', 'avoid_public_places', 'maintain_distance', 'adjust_school_work', 'quarantine_symptoms',
         'quarantine_no_symptoms']
    ].sum(axis='columns').astype(int)

    compliance['Month'] = compliance.index.get_level_values('month').month_name()
    compliance.to_pickle(produces)

@pytask.mark.depends_on(
    {
        month: SRC / "original_data" / "liss" / f"covid_data_{month}.pickle"
        for month in ["2020_05", "2020_06", "2020_09", "2020_12"]
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "work_status.pickle")
def task_clean_work_status_data(depends_on, produces):

    covid_data_list = [pd.read_pickle(wave) for wave in depends_on.values()]
    covid_data = pd.concat(covid_data_list)
    select_covid = covid_data[["work_status"]].dropna(axis=0, how='any')

    select_covid['Month'] = select_covid.index.get_level_values('month').month_name()
    select_covid.to_pickle(produces)

@pytask.mark.depends_on(
    {
        month: SRC / "original_data" / "liss" / f"covid_data_{month}.pickle"
        for month in ["2020_04"]
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "essential_worker.pickle")
def task_clean_essential_worker_data(depends_on, produces):

    covid_data_list = [pd.read_pickle(wave) for wave in depends_on.values()]
    covid_data = pd.concat(covid_data_list)
    select_covid = covid_data[["essential_worker"]].dropna(axis=0, how='any')

    select_covid['Month'] = select_covid.index.get_level_values('month').month_name()
    select_covid.to_pickle(produces)
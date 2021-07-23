import numpy as np
import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(
    SRC / "original_data" / "liss" / "background_data_merged.pickle"
)
@pytask.mark.produces(BLD / "data" / "liss" / "background.pickle")
def task_clean_background_data(depends_on, produces):
    background = pd.read_pickle(depends_on)
    select_background = background[
        [
            "hh_id",
            "age",
            "female",
            "edu",
            "edu_4",
            "net_income",
            "hh_members",
            "hh_children",
            "net_income_hh_eqv",
            "income_hh_group",
            "net_income_hh",
            "gross_income_hh",
        ]
    ].dropna(axis=0, how="any")
    select_background["age_cut"] = pd.cut(
        select_background.age,
        [15, 25, 50, 75, 120],
        right=False,
        labels=["<25", "25-50", "50-75", ">75"],
    )
    select_background["living_alone"] = (select_background["hh_members"] == 1).astype(
        int
    )  # alone dummy 1: alone, 0: not alone
    select_background["living_with_children"] = (
        select_background["hh_children"] > 0
    ).astype(
        int
    )  # 1: has children, 0: no children
    select_background["age"] = select_background["age"].astype("int")
    select_background["age_by100"] = select_background["age"] / 100
    select_background["hh_id"] = select_background["hh_id"].astype("int")
    select_background["female"] = select_background["female"].astype("int")
    select_background["log_income_hh"] = np.log(
        select_background["net_income_hh_eqv"].clip(lower=0.001)
    )
    select_background["income_hh_cut"] = pd.cut(
        select_background["net_income_hh_eqv"],
        select_background["net_income_hh_eqv"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]),
        right=False,
        labels=["lowest", "low", "middle", "high", "highest"],
    )
    select_background["net_income_hh_eqv"] = (
        select_background["net_income_hh_eqv"].clip(
            upper=select_background["net_income_hh_eqv"].quantile(0.99)
        )
        / 1000
    )
    select_background["net_income_hh"] = (
        select_background["net_income_hh"].clip(
            upper=select_background["net_income_hh"].quantile(0.99)
        )
        / 1000
    )
    select_background["gross_income_hh"] = (
        select_background["gross_income_hh"].clip(
            upper=select_background["gross_income_hh"].quantile(0.99)
        )
        / 1000
    )
    select_background["male"] = 1 - select_background["female"]
    select_background["edu"].cat.rename_categories(
        {
            "lower_secondary_and_lower": "lower sec. & less",
            "upper_secondary": "upper sec.",
            "tertiary": "tertiary",
        },
        inplace=True,
    )

    select_background.rename(columns={})

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
    select_covid = covid_data[["infection_diagnosed", "infection_perceived"]].dropna(
        axis=0, how="all"
    )
    select_covid["infection_diagnosed"] = (
        select_covid["infection_diagnosed"]
        .cat.rename_categories({"yes, I have been diagnosed with it": "yes"})
        .cat.reorder_categories(["no", "unsure", "yes"], ordered=True)
    )
    # 1: infection diagnosed, 0: not infected diagnosed or unsure
    select_covid["infected"] = select_covid["infection_diagnosed"].eq("yes").astype(int)

    select_covid["Month"] = select_covid.index.get_level_values("month").month_name()

    infected_wide = select_covid["infection_diagnosed"].cat.codes.unstack()
    for i in range(infected_wide.shape[0]):
        for j in range(infected_wide.shape[1] - 1, 0, -1):
            if infected_wide.iloc[i, j] <= infected_wide.iloc[i, j - 1]:
                infected_wide.iloc[i, j] = 0
    new_infected = (
        infected_wide.stack()
        .rename("new_infected")
        .astype("category")
        .cat.rename_categories(["no", "unsure", "yes"])
        .cat.reorder_categories(["no", "unsure", "yes"], ordered=True)
    )
    select_covid = select_covid.join(new_infected)
    select_covid.to_pickle(produces)


@pytask.mark.depends_on(
    {
        "covid_data": SRC / "original_data" / "liss" / "covid_data_2020_03.pickle",
        "background": BLD / "data" / "liss" / "background.pickle",
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "compliance.pickle")
def task_clean_compliance_data(depends_on, produces):
    covid_data = pd.read_pickle(depends_on["covid_data"])
    compliance = covid_data[
        [
            "avoid_busy_places",
            "avoid_public_places",
            "maintain_distance",
            "adjust_school_work",
            "quarantine_symptoms",
            "quarantine_no_symptoms",
            "no_avoidance_behaviors",
            "comply_curfew_self",
        ]
    ].dropna(axis=0, how="any")
    # compliance_index is sum of these compliance variables
    compliance["compliance_index"] = 0
    loc = (compliance["no_avoidance_behaviors"] != 1) & (
        compliance["comply_curfew_self"] != "no"
    )
    compliance.loc[loc, "compliance_index"] = (
        compliance.loc[
            loc,
            [
                "avoid_busy_places",
                "avoid_public_places",
                "maintain_distance",
                "adjust_school_work",
                "quarantine_symptoms",
                "quarantine_no_symptoms",
            ],
        ]
        # .dot(pd.Series({"avoid_busy_places": 1,
        #                 "avoid_public_places": 1,
        #                 "maintain_distance": 1,
        #                 "adjust_school_work": 2,
        #                 "quarantine_symptoms": 5,
        #                 "quarantine_no_symptoms": 6}))
        .sum(axis="columns").astype(int)
    )

    compliance["Month"] = compliance.index.get_level_values("month").month_name()

    # merge background['hh_id']
    background = pd.read_pickle(depends_on["background"])
    merge_data = pd.merge(
        compliance.reset_index("month"),
        background[["hh_id", "hh_members"]],
        on="personal_id",
        how="inner",
    )

    # compliance_index_hh = sum of compliance_index in same household/ hh_members
    compliance_index_hh = (
        merge_data.groupby("hh_id")
        .sum()["compliance_index"]
        .rename("compliance_index_hh")
    )
    merge_data = merge_data.join(compliance_index_hh, on="hh_id", how="inner")
    merge_data["compliance_index_hh"] = (
        merge_data["compliance_index_hh"] / merge_data["hh_members"]
    )

    # # compliance_index_hh = min of compliance_index in same household
    # compliance_index_hh = (
    #     merge_data.groupby("hh_id")
    #     .min()["compliance_index"]
    #     .rename("compliance_index_hh")
    # )
    # merge_data = merge_data.join(compliance_index_hh, on="hh_id", how="inner")

    merge_data.set_index("month", append=True, inplace=True)
    merge_data.drop(columns=["hh_id", "hh_members"], inplace=True)
    merge_data.to_pickle(produces)


@pytask.mark.depends_on(
    {
        month: SRC / "original_data" / "liss" / f"covid_data_{month}.pickle"
        for month in ["2020_02", "2020_05", "2020_06", "2020_09", "2020_12"]
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "work_status.pickle")
def task_clean_work_status_data(depends_on, produces):
    covid_data_list = [pd.read_pickle(wave) for wave in depends_on.values()]
    covid_data = pd.concat(covid_data_list).reset_index("month")
    month = covid_data["month"].replace(
        pd.Timestamp(year=2020, month=2, day=1), pd.Timestamp(year=2020, month=4, day=1)
    )
    covid_data["month"] = month
    select_covid = covid_data.set_index(["month"], append=True)[["work_status"]].dropna(
        axis=0, how="any"
    )

    select_covid.loc[:, "occupation"] = pd.Categorical(
        select_covid["work_status"].replace(
            {
                "retired": "unemployed",
                "homemaker": "unemployed",
                "student or trainee": "unemployed",
                "social assistance": "unemployed",
                "self-employed": "employed",
                "unemployed": "unemployed",
                "employed": "employed",
            }
        )
    ).reorder_categories(["unemployed", "employed"])
    select_covid["employed"] = select_covid["occupation"].eq("employed").astype(int)
    select_covid["Month"] = select_covid.index.get_level_values("month").month_name()

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
    select_covid = covid_data[["essential_worker"]].dropna(axis=0, how="any")

    select_covid["Month"] = select_covid.index.get_level_values("month").month_name()
    select_covid.to_pickle(produces)


@pytask.mark.depends_on(SRC / "original_data" / "liss" / "cw19l_EN_3.0p.dta")
@pytask.mark.produces(BLD / "data" / "liss" / "industry.pickle")
def task_clean_industry_data(depends_on, produces):
    work_schooling = pd.read_stata(depends_on)
    industry_data = work_schooling[["nomem_encr", "cw19l_m", "cw19l402"]]
    industry_data.insert(
        0,
        "month",
        pd.to_datetime(industry_data["cw19l_m"], format="%Y%m", errors="ignore"),
    )
    industry_data.insert(0, "personal_id", industry_data["nomem_encr"].astype(int))
    industry_data = (
        industry_data.set_index(["personal_id", "month"])["cw19l402"]
        .rename("industry")
        .dropna()
    )

    industry_data.to_pickle(produces)

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
    loc = (
        compliance["no_avoidance_behaviors"]
        != 1
        #    ) & (
        # compliance["comply_curfew_self"] != "no"
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
        .dot(
            pd.Series(
                {
                    "avoid_busy_places": 1,
                    "avoid_public_places": 1,
                    "maintain_distance": 1,
                    "adjust_school_work": 1,
                    "quarantine_symptoms": 1,
                    "quarantine_no_symptoms": 1,
                }
            )
        )
        .astype(int)
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
        "covid03": SRC / "original_data" / "liss" / "covid_data_2020_03.pickle",
        "covid04": SRC / "original_data" / "liss" / "covid_data_2020_04.pickle",
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "essential_worker.pickle")
def task_clean_essential_worker_data(depends_on, produces):
    cruciaal = (
        pd.read_pickle(depends_on["covid04"])
        .reset_index("month")["essential_worker"]
        .dropna()
    )
    qualify_essential_worker = cruciaal.astype(int).rename("qualify_essential_worker")
    q10 = (
        pd.read_pickle(depends_on["covid03"])
        .reset_index("month")["comply_curfew_self"]
        .dropna()
    )
    working_essential_worker = (
        q10.eq("critical profession").astype("int").rename("working_essential_worker")
    )

    essential_worker = pd.merge(
        qualify_essential_worker,
        working_essential_worker,
        on="personal_id",
        how="outer",
    )
    essential_worker.to_pickle(produces)


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


@pytask.mark.depends_on(SRC / "original_data" / "liss" / "cp21m_EN_1.0p.dta")
@pytask.mark.produces(BLD / "data" / "liss" / "personality.pickle")
def task_clean_personality_data(depends_on, produces):
    ori_data = pd.read_stata(depends_on)
    ori_data.insert(0, "personal_id", ori_data["nomem_encr"].astype(int))
    ori_data.set_index(["personal_id"], inplace=True)

    big_5 = ori_data[["cp21m0" + str(i) for i in range(20, 70)]].rename(
        columns={
            "cp21m020": "am the life of the party",
            "cp21m021": "feel little concern for others",
            "cp21m022": "am always prepared",
            "cp21m023": "get stressed out easily",
            "cp21m024": "have a rich vocabulary",
            "cp21m025": "don't talk a lot",
            "cp21m026": "am interested in people",
            "cp21m027": "leave my belongings around",
            "cp21m028": "am relaxed most of the time",
            "cp21m029": "have difficulty understanding abstract ideas",
            "cp21m030": "feel comfortable around people",
            "cp21m031": "insult people",
            "cp21m032": "pay attention to details",
            "cp21m033": "worry about things",
            "cp21m034": "have a vivid imagination",
            "cp21m035": "keep in the background",
            "cp21m036": "sympathize with others' feelings",
            "cp21m037": "make a mess of things",
            "cp21m038": "seldom feel blue",
            "cp21m039": "am not interested in abstract ideas",
            "cp21m040": "start conversations",
            "cp21m041": "am not interested in other people's problems",
            "cp21m042": "get chores done right away",
            "cp21m043": "am easily disturbed",
            "cp21m044": "have excellent ideas",
            "cp21m045": "have little to say",
            "cp21m046": "have a soft heart",
            "cp21m047": "often forget to put things back in their proper place",
            "cp21m048": "get upset easily",
            "cp21m049": "do not have a good imagination",
            "cp21m050": "talk to a lot of different people at parties",
            "cp21m051": "am not really interested in others",
            "cp21m052": "like order",
            "cp21m053": "change my mood a lot",
            "cp21m054": "am quick to understand things",
            "cp21m055": "don't like to draw attention to myself",
            "cp21m056": "take time out for others",
            "cp21m057": "shirk my duties",
            "cp21m058": "have frequent mood swings",
            "cp21m059": "use difficult words",
            "cp21m060": "don't mind being the center of attention",
            "cp21m061": "feel others' emotions",
            "cp21m062": "follow a schedule",
            "cp21m063": "get irritated easily",
            "cp21m064": "spend time reflecting on things",
            "cp21m065": "am quiet around strangers",
            "cp21m066": "make people feel at ease",
            "cp21m067": "am exacting in my work",
            "cp21m068": "often feel blue",
            "cp21m069": "am full of ideas",
        }
    )

    extraversion = (
        big_5["am the life of the party"].cat.codes
        + big_5["feel comfortable around people"].cat.codes
        + big_5["start conversations"].cat.codes
        + big_5["talk to a lot of different people at parties"].cat.codes
        + big_5["don't mind being the center of attention"].cat.codes
        - big_5["don't talk a lot"].cat.codes
        - big_5["keep in the background"].cat.codes
        - big_5["have little to say"].cat.codes
        - big_5["don't like to draw attention to myself"].cat.codes
        - big_5["am quiet around strangers"].cat.codes
    )
    openness = (
        big_5["have a rich vocabulary"].cat.codes
        + big_5["have a vivid imagination"].cat.codes
        + big_5["have excellent ideas"].cat.codes
        + big_5["am quick to understand things"].cat.codes
        + big_5["use difficult words"].cat.codes
        + big_5["spend time reflecting on things"].cat.codes
        + big_5["am full of ideas"].cat.codes
        - big_5["have difficulty understanding abstract ideas"].cat.codes
        - big_5["am not interested in abstract ideas"].cat.codes
        - big_5["do not have a good imagination"].cat.codes
    )
    conscientiousness = (
        big_5["am always prepared"].cat.codes
        + big_5["pay attention to details"].cat.codes
        + big_5["get chores done right away"].cat.codes
        + big_5["like order"].cat.codes
        + big_5["follow a schedule"].cat.codes
        + big_5["am exacting in my work"].cat.codes
        - big_5["leave my belongings around"].cat.codes
        - big_5["make a mess of things"].cat.codes
        - big_5["often forget to put things back in their proper place"].cat.codes
        - big_5["shirk my duties"].cat.codes
    )
    agreeableness = (
        big_5["am interested in people"].cat.codes
        + big_5["sympathize with others' feelings"].cat.codes
        + big_5["have a soft heart"].cat.codes
        + big_5["take time out for others"].cat.codes
        + big_5["feel others' emotions"].cat.codes
        + big_5["make people feel at ease"].cat.codes
        - big_5["am not really interested in others"].cat.codes
        - big_5["insult people"].cat.codes
        - big_5["am not interested in other people's problems"].cat.codes
        - big_5["feel little concern for others"].cat.codes
    )
    neuroticism = (
        -big_5["am relaxed most of the time"].cat.codes
        - big_5["seldom feel blue"].cat.codes
        + big_5["get stressed out easily"].cat.codes
        + big_5["worry about things"].cat.codes
        + big_5["am easily disturbed"].cat.codes
        + big_5["get upset easily"].cat.codes
        + big_5["change my mood a lot"].cat.codes
        + big_5["have frequent mood swings"].cat.codes
        + big_5["get irritated easily"].cat.codes
        + big_5["often feel blue"].cat.codes
    )

    personality = pd.DataFrame(
        {
            "extraversion": extraversion,
            "openness": openness,
            "conscientiousness": conscientiousness,
            "agreeableness": agreeableness,
            "neuroticism": neuroticism,
        }
    )
    personality.to_pickle(produces)


@pytask.mark.depends_on(SRC / "original_data" / "liss" / "cv21m_EN_1.0p.dta")
@pytask.mark.produces(BLD / "data" / "liss" / "politics.pickle")
def task_clean_politics_data(depends_on, produces):
    ori_data = pd.read_stata(depends_on)
    ori_data.insert(0, "personal_id", ori_data["nomem_encr"].astype(int))
    ori_data.set_index(["personal_id"], inplace=True)

    # 0 ("Left") to 10 ("Right").
    ideology = (
        ori_data["cv21m101"]
        .rename("ideology")
        .replace("I dont know", np.nan)
        .cat.remove_unused_categories()
        .cat.codes
    )

    politics = pd.DataFrame({"ideology": ideology})
    politics.to_pickle(produces)


@pytask.mark.depends_on(
    {
        "god": SRC / "original_data" / "liss" / "mb15a_EN_1.1p.dta",
        "covid03": SRC / "original_data" / "liss" / "covid_data_2020_03.pickle",
    }
)
@pytask.mark.produces(BLD / "data" / "liss" / "trust.pickle")
def task_trust_data(depends_on, produces):
    god_data = pd.read_stata(depends_on["god"])
    god_data.insert(0, "personal_id", god_data["nomem_encr"].astype(int))
    god_data.set_index(["personal_id"], inplace=True)
    covid03_data = pd.read_pickle(depends_on["covid03"]).reset_index("month")

    # 1 = no trust at all 10 = complete trust
    trust_institutions = (
        god_data[["mb15a011", "mb15a013"]]
        .rename(
            columns={"mb15a011": "trust_science", "mb15a013": "trust_political_parties"}
        )
        .replace("don't know / no opinion", np.nan)
    )
    trust_institutions["trust_science"] = (
        trust_institutions["trust_science"].cat.remove_unused_categories().cat.codes / 2
    )
    trust_institutions["trust_political_parties"] = (
        trust_institutions["trust_political_parties"]
        .cat.remove_unused_categories()
        .cat.codes
        / 2
    )

    trust_gov = (
        covid03_data["trust_gov"]
        .cat.remove_unused_categories()
        .cat.codes.rename("trust_gov")
    )

    trust = pd.merge(
        trust_institutions, trust_gov, on="personal_id", how="outer"
    ).dropna(how="all")
    trust.to_pickle(produces)

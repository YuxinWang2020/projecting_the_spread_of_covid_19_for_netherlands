import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(SRC / "original_data" / "owid-covid-data.csv")
@pytask.mark.produces(BLD / "data" / "covid_SIRD.pickle")
def task_OxCGRT_policy_stringency(depends_on, produces):
    owid = pd.read_csv(depends_on)
    owid = owid.query("location == 'Netherlands'")
    owid = owid.set_index(pd.to_datetime(owid["date"]))
    covid_SIRD = (
        owid[["total_cases", "total_deaths"]]
        .groupby(pd.Grouper(freq="M"))
        .mean()
        .round(0)
    )
    covid_SIRD = covid_SIRD.set_axis(
        covid_SIRD.index.to_period("M").to_timestamp().rename("month"), axis="index"
    )
    covid_SIRD.to_pickle(produces)

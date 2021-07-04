import pandas as pd
import pytask

from src.config import BLD


@pytask.mark.depends_on(
    {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle",
        "infected": BLD / "data" / "liss" / "infected.pickle",
        "work_status": BLD / "data" / "liss" / "work_status.pickle",
    }
)
@pytask.mark.produces(BLD / "tables" / "stat_infected_x_var.csv")
def task_stat_infected_x_var(depends_on, produces):

    compliance = pd.read_pickle(depends_on["compliance"])[
        "compliance_index"
    ].reset_index(level="month", drop=True)
    infected = pd.read_pickle(depends_on["infected"])
    work_status = pd.read_pickle(depends_on["work_status"]).drop(columns="Month")
    background = pd.read_pickle(depends_on["background"])
    merge_data = (
        infected.join(background, on="personal_id", how="inner")
        .join(compliance, on="personal_id", how="inner")
        .join(work_status, on=["personal_id", "month"], how="inner")
        .reset_index(["month"])
    )

    x = merge_data[["female", "living_alone", "living_with_children"]]

    # change category to dummy
    x = x.join(
        pd.get_dummies(
            merge_data["edu"].cat.remove_unused_categories(),
            prefix="edu",
            drop_first=False,
            prefix_sep=":",
        )
    )
    x = x.join(
        pd.get_dummies(
            merge_data["age_cut"], prefix="age", drop_first=False, prefix_sep=":"
        )
    )
    x = x.join(
        pd.get_dummies(
            merge_data["occupation"],
            prefix="occupation",
            drop_first=False,
            prefix_sep=":",
        )
    )
    x = x.join(
        pd.get_dummies(
            merge_data["income_hh_cut"],
            prefix="income_hh",
            drop_first=False,
            prefix_sep=":",
        )
    )

    x_var = (
        x.reset_index(level="personal_id")
        .drop_duplicates(inplace=False)
        .drop(columns="personal_id")
    )
    df_desc_stat = pd.DataFrame(
        {
            "Observations": x_var.count(),
            "Mean": x_var.mean(),
            "Std": x_var.std(),
            "Min": x_var.min(),
            "Max": x_var.max(),
            "Sum": x_var.sum(),
        }
    )
    df_desc_stat.to_csv(produces, float_format="%.3f")


@pytask.mark.depends_on(
    {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle",
        "infected": BLD / "data" / "liss" / "infected.pickle",
        "work_status": BLD / "data" / "liss" / "work_status.pickle",
    }
)
@pytask.mark.produces(BLD / "tables" / "stat_infected_y_var.csv")
def task_stat_infected_y_var(depends_on, produces):
    compliance = pd.read_pickle(depends_on["compliance"])[
        "compliance_index"
    ].reset_index(level="month", drop=True)
    infected = pd.read_pickle(depends_on["infected"])
    work_status = pd.read_pickle(depends_on["work_status"]).drop(columns="Month")
    background = pd.read_pickle(depends_on["background"])
    merge_data = (
        infected.join(background, on="personal_id", how="inner")
        .join(compliance, on="personal_id", how="inner")
        .join(work_status, on=["personal_id", "month"], how="inner")
        .reset_index(["month"])
    )
    y_var = merge_data[["infected", "month", "Month"]]
    stat_y = (
        pd.DataFrame(
            {
                "Observations": y_var.groupby(["Month", "month"]).count()["infected"],
                "Total infected": y_var.groupby(["Month", "month"]).sum()["infected"],
            }
        )
        .sort_index(level="month")
        .reset_index("month", drop=True)
    )
    stat_y["Infected rate"] = stat_y["Total infected"] / stat_y["Observations"]
    stat_y.to_csv(produces, float_format="%.3f")

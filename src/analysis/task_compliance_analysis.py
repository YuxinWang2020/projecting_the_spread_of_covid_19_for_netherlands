import pandas as pd
import pytask

from src.config import BLD
from src.model_code.format_result import odds_radio_format
from src.model_code.format_result import sm_results_format
from src.model_code.regression import ols_regression_formula
from src.model_code.regression import ordinal_logit_regression_formula


@pytask.mark.depends_on(
    {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle",
        "work_status": BLD / "data" / "liss" / "work_status.pickle",
    }
)
@pytask.mark.produces(BLD / "tables" / "stat_compliance_x_var.csv")
def task_stat_compliance_x_var(depends_on, produces):
    compliance = (
        pd.read_pickle(depends_on["compliance"])
        .reset_index(level="month", drop=True)
        .drop(columns="Month")
    )
    work_status = (
        pd.read_pickle(depends_on["work_status"])
        .query("month == '2020-04-01'")
        .reset_index(level="month", drop=True)
        .drop(columns="Month")
    )
    background = pd.read_pickle(depends_on["background"])
    merge_data = compliance.join(background, on="personal_id", how="inner").join(
        work_status, on="personal_id", how="inner"
    )

    # x_var = dmatrix(
    #     "female + living_alone + living_with_children + age_cut + edu + occupation + income_hh_cut"
    #     " + income_hh_cut*living_alone",
    #     merge_data,
    #     return_type="dataframe",
    # ).drop(columns="Intercept")

    x_var = merge_data[["female", "living_alone", "living_with_children"]]

    # change category to dummy
    x_var = x_var.join(
        pd.get_dummies(
            merge_data["edu"].cat.remove_unused_categories(),
            prefix="edu",
            drop_first=False,
            prefix_sep=":",
        )
    )
    x_var = x_var.join(
        pd.get_dummies(
            merge_data["age_cut"], prefix="age", drop_first=False, prefix_sep=":"
        )
    )
    x_var = x_var.join(
        pd.get_dummies(
            merge_data["occupation"],
            prefix="occupation",
            drop_first=False,
            prefix_sep=":",
        )
    )
    x_var = x_var.join(
        pd.get_dummies(
            merge_data["income_hh_cut"],
            prefix="income_hh",
            drop_first=False,
            prefix_sep=":",
        )
    )

    x_var = (
        x_var.reset_index(level="personal_id")
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
    }
)
@pytask.mark.produces(BLD / "tables" / "stat_compliance_y_var.csv")
def task_stat_compliance_y_var(depends_on, produces):
    compliance = pd.read_pickle(depends_on["compliance"]).reset_index(["month"])
    background = pd.read_pickle(depends_on["background"])
    merge_data = compliance.join(background, on="personal_id", how="inner")
    y_var = merge_data[["compliance_index", "month", "Month"]]
    stat_y = (
        pd.DataFrame(
            {
                "Observations": y_var.groupby(["Month", "month"]).count()[
                    "compliance_index"
                ],
                "Total": y_var.groupby(["Month", "month"]).sum()["compliance_index"],
                "Mean": y_var.groupby(["Month", "month"]).mean()["compliance_index"],
                "Std": y_var.groupby(["Month", "month"]).std()["compliance_index"],
                "Min": y_var.groupby(["Month", "month"]).min()["compliance_index"],
                "Max": y_var.groupby(["Month", "month"]).max()["compliance_index"],
            }
        )
        .sort_index(level="month")
        .reset_index("month", drop=True)
    )
    stat_y.to_csv(produces, float_format="%.3f")


@pytask.mark.depends_on(
    {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle",
        "work_status": BLD / "data" / "liss" / "work_status.pickle",
    }
)
@pytask.mark.produces(
    {
        "regression": BLD / "tables" / "compliance_ordered_logit.csv",
        "odds_radio": BLD / "tables" / "compliance_ordered_logit_OR.csv",
    }
)
def task_compliance_ordinal_regression(depends_on, produces):
    # merge data
    compliance = (
        pd.read_pickle(depends_on["compliance"])
        .reset_index(level="month", drop=True)
        .drop(columns="Month")
    )
    work_status = (
        pd.read_pickle(depends_on["work_status"])
        .query("month == '2020-04-01'")
        .reset_index(level="month", drop=True)
        .drop(columns="Month")
    )
    background = pd.read_pickle(depends_on["background"])
    merge_data = compliance.join(background, on="personal_id", how="inner").join(
        work_status, on="personal_id", how="inner"
    )

    # run regression
    formula = (
        "compliance_index ~ female + living_alone + living_with_children + age + I(age**2) + "
        "edu + occupation + net_income_hh_eqv"
        " + female*edu"
    )
    ordinal_result, _, ordinal_odds_radio = ordinal_logit_regression_formula(
        merge_data, formula
    )
    ols_result, _ = ols_regression_formula(merge_data, formula)

    formed_odds_radios = odds_radio_format([ordinal_odds_radio], ["Odds Ratio"])
    formed_result = sm_results_format(
        [ordinal_result, ols_result], ["Ordered Logit", "OLS"]
    )

    # with open(produces['regression'], 'w') as f:
    #     f.write(formed_result.as_latex())
    pd.DataFrame(formed_result.tables[0]).to_csv(
        produces["regression"], float_format="%.3f"
    )
    formed_odds_radios.to_csv(produces["odds_radio"], float_format="%.3f")


def _get_compliance_XY(merge_data):
    y = merge_data.loc[:, "compliance_index"]
    x = merge_data[["female", "living_alone", "living_with_children"]]

    # change category to dummy
    x = x.join(
        pd.get_dummies(
            merge_data["edu"].cat.remove_unused_categories(),
            prefix="edu",
            drop_first=True,
            prefix_sep=":",
        )
    )
    x = x.join(
        pd.get_dummies(
            merge_data["age_cut"], prefix="age", drop_first=True, prefix_sep=":"
        )
    )

    # add interaction
    # x['edu:tertiary # age:[50, 75)'] = x['age:[50, 75)'] * x['edu:tertiary']
    # x['living_alone # age:[25, 50)'] = x['age:[25, 50)'] * x['living_alone']
    return x, y

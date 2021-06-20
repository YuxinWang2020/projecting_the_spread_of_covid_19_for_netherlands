import pandas as pd
import pytask

from src.config import BLD
from src.model_code.format_result import odds_radio_format
from src.model_code.format_result import sm_results_format
from src.model_code.regression import add_interaction
from src.model_code.regression import binomial_logit_regression
from src.model_code.regression import binomial_logit_regression_formula


@pytask.mark.depends_on(
    {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle",
        "infected": BLD / "data" / "liss" / "infected.pickle",
        "work_status": BLD / "data" / "liss" / "work_status.pickle",
    }
)
@pytask.mark.produces(
    {
        "regression": BLD / "tables" / "infected_with_compliance_logit.csv",
        "odds_radio": BLD / "tables" / "infected_with_compliance_logit_OR.csv",
    }
)
def task_infected_with_compliance_binomial_regression(depends_on, produces):
    compliance = pd.read_pickle(depends_on["compliance"])[
        "compliance_index"
    ].reset_index(level="month", drop=True)
    infected = pd.read_pickle(depends_on["infected"])
    work_status = pd.read_pickle(depends_on["work_status"])
    background = pd.read_pickle(depends_on["background"])
    merge_data = (
        infected.join(background, on="personal_id", how="inner")
        .join(compliance, on="personal_id", how="inner")
        .join(work_status, on=["personal_id", "month"], how="inner")
    )
    merge_data = merge_data.sort_index(level=["personal_id", "month"])

    model_names = (
        merge_data.index.get_level_values("month")
        .drop_duplicates()
        .sort_values()
        .month_name()
        .tolist()
    )
    results = []
    odds_radios = []
    for month in model_names:  # noqa:B007
        merge_data_month = merge_data.query("Month == @month")
        result, summary, odds_radio = _infected_binomial_regression_formula(
            merge_data_month
        )
        results.append(result)
        odds_radios.append(odds_radio)

    result, summary, odds_radio = _infected_binomial_regression_formula(merge_data)
    model_names.append("Pooled")
    results.append(result)
    odds_radios.append(odds_radio)

    formated_result = sm_results_format(results, model_names)
    formated_odds_radios = odds_radio_format(odds_radios, model_names)

    pd.DataFrame(formated_result.tables[0]).to_csv(
        produces["regression"], float_format="%.3f"
    )
    # with open(produces['regression'], 'w') as f:
    #     f.write(formated_result.as_latex())
    formated_odds_radios.to_csv(produces["odds_radio"], float_format="%.3f")


def _infected_binomial_regression(merge_data):
    y = merge_data["infected"]
    x = merge_data[
        ["compliance_index", "female", "living_alone", "living_with_children"]
    ]

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
    # add_interaction(x, 'age:[50, 75)', 'edu:tertiary')
    add_interaction(x, "compliance_index", "edu:upper_secondary")
    add_interaction(x, "compliance_index", "edu:tertiary")
    # add_interaction(x, 'compliance_index', 'age:[25, 50)', 'living_alone')

    # run regression
    result, summary, odds_radio = binomial_logit_regression(x, y)
    return result, summary, odds_radio


def _infected_binomial_regression_formula(merge_data):

    formula = (
        "infected ~ compliance_index + female + living_alone + living_with_children + age_cut + "
        "edu + occupation + income_hh_group"
    )

    # add interaction

    # x['edu:tertiary # age:[50, 75)'] = x['age:[50, 75)'] * x['edu:tertiary']
    # x['living_alone # age:[25, 50)'] = x['age:[25, 50)'] * x['living_alone']
    # add_interaction(x, 'age:[50, 75)', 'edu:tertiary')
    # add_interaction(x, 'compliance_index', 'age:[25, 50)', 'living_alone')

    # run regression
    result, summary, odds_radio = binomial_logit_regression_formula(merge_data, formula)
    return result, summary, odds_radio

import pandas as pd
import pytask

from src.model_code.discrete_regression import ordinal_logit_regression
from src.model_code.format_result import sm_results_format
from src.model_code.format_result import odds_radio_format

from src.config import BLD

@pytask.mark.depends_on({
    "background": BLD / "data" / "liss" / "background.pickle",
    "compliance": BLD / "data" / "liss" / "compliance.pickle"
})
@pytask.mark.produces(BLD / "tables" / "stat_compliance_x_var.csv")
def task_stat_compliance_x_var(depends_on, produces):
    depends_on = {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle"
    }
    produces = BLD / "tables" / "stat_compliance_x_var.csv"

    compliance = pd.read_pickle(depends_on['compliance']).reset_index(['month'])
    background = pd.read_pickle(depends_on['background'])
    merge_data = compliance.join(background, on="personal_id", how="inner")

    x = merge_data[['female', 'living_alone', 'living_with_children']]

    # change category to dummy
    x = x.join(pd.get_dummies(merge_data['edu'].cat.remove_unused_categories(), prefix='edu', drop_first=False,
                              prefix_sep=':'))
    x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age', drop_first=False, prefix_sep=':'))

    x_var = x.reset_index(level='personal_id').drop_duplicates(inplace=False).drop(columns='personal_id')
    df_desc_stat = pd.DataFrame({
        "Observations": x_var.count(),
        "Mean": x_var.mean(),
        "Std": x_var.std(),
        "Min": x_var.min(),
        "Max": x_var.max(),
        "Sum": x_var.sum()
    })
    df_desc_stat.to_csv(produces, float_format="%.3f")


@pytask.mark.depends_on({
    "background": BLD / "data" / "liss" / "background.pickle",
    "compliance": BLD / "data" / "liss" / "compliance.pickle"
})
@pytask.mark.produces(BLD / "tables" / "stat_compliance_y_var.csv")
def task_stat_compliance_y_var(depends_on, produces):
    depends_on = {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle"
    }
    produces = BLD / "tables" / "stat_compliance_y_var.csv"

    compliance = pd.read_pickle(depends_on['compliance']).reset_index(['month'])
    background = pd.read_pickle(depends_on['background'])
    merge_data = compliance.join(background, on="personal_id", how="inner")
    y_var = merge_data[['compliance_index', 'month', 'Month']]
    stat_y = pd.DataFrame({
        "Observations": y_var.groupby(['Month','month']).count()['compliance_index'],
        "Total": y_var.groupby(['Month','month']).sum()['compliance_index'],
        "Mean": y_var.groupby(['Month','month']).mean()['compliance_index'],
        "Std": y_var.groupby(['Month','month']).std()['compliance_index'],
        "Min": y_var.groupby(['Month','month']).min()['compliance_index'],
        "Max": y_var.groupby(['Month','month']).max()['compliance_index'],
    }).sort_index(level='month').reset_index('month',drop=True)
    stat_y.to_csv(produces, float_format="%.3f")


@pytask.mark.depends_on({
    "background": BLD / "data" / "liss" / "background.pickle",
    "compliance": BLD / "data" / "liss" / "compliance.pickle"
})
@pytask.mark.produces({
    "regression": BLD / "tables" / "compliance_ordered_logit.csv",
    "odds_radio": BLD / "tables" / "compliance_ordered_logit_OR.csv"
})
def task_compliance_ordinal_regression(depends_on, produces):
    depends_on = {
        "background": BLD / "data" / "liss" / "background.pickle",
        "compliance": BLD / "data" / "liss" / "compliance.pickle"
    }
    produces = {
        "regression": BLD / "tables" / "compliance_ordered_logit.csv",
        "odds_radio": BLD / "tables" / "compliance_ordered_logit_OR.csv"
    }

    compliance = pd.read_pickle(depends_on['compliance'])
    background = pd.read_pickle(depends_on['background'])
    merge_data = compliance.join(background, on="personal_id", how="inner")
    merge_data = merge_data.sort_index(level = ['personal_id','month'])

    model_names = merge_data['Month'].drop_duplicates().tolist()
    results = []
    odds_radios = []
    for month in model_names:
        merge_data_month = merge_data.query("Month == @month")
        result, summary, odds_radio = _compliance_ordinal_regression(merge_data_month)
        results.append(result)
        odds_radios.append(odds_radio)

    formated_odds_radios = odds_radio_format(odds_radios, model_names)
    formated_result = sm_results_format(results, model_names)

    # formated_result.as_latex(produces['regression'], float_format="%.3f")
    with open(produces['regression'], 'w') as f:
        f.write(formated_result.as_latex())
    formated_odds_radios.to_csv(produces['odds_radio'], float_format="%.3f")


def _compliance_ordinal_regression(merge_data):
    y = merge_data.loc[:, "compliance_index"]
    x = merge_data[['female', 'living_alone', 'living_with_children']]

    # change category to dummy
    x = x.join(pd.get_dummies(merge_data['edu'].cat.remove_unused_categories(), prefix='edu', drop_first=True,
                              prefix_sep=':'))
    x = x.join(pd.get_dummies(merge_data['age_cut'], prefix='age', drop_first=True, prefix_sep=':'))

    # add interaction
    # x['edu:tertiary # age:[50, 75)'] = x['age:[50, 75)'] * x['edu:tertiary']
    x['living_alone # age:[25, 50)'] = x['age:[25, 50)'] * x['living_alone']

    # run regression
    result, summary, odds_radio = ordinal_logit_regression(x, y)
    return result, summary, odds_radio

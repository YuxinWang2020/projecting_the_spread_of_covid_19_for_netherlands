import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


def binomial_logit_regression(x, y, intercept=True):
    if intercept:
        x = sm.add_constant(x)  # add constant if need intercept
    # run regression
    model = sm.Logit(y, x)
    result = model.fit()
    summary = result.summary()
    odds_radio = get_odds_radio(result)
    return result, summary, odds_radio


def binomial_logit_regression_formula(data, formula):
    model = sm.Logit.from_formula(formula=formula, data=data)
    result = model.fit()
    summary = result.summary()
    odds_radio = get_odds_radio(result)
    return result, summary, odds_radio


def get_odds_radio(result):
    # get Odds Ratio
    conf = result.conf_int()
    conf["Odds Ratio"] = result.params
    conf.columns = ["5%", "95%", "Odds Ratio"]
    return np.exp(conf)


def ordinal_logit_regression(x, y):
    model = OrderedModel(y, x, distr="logit")
    result = model.fit(method="bfgs")
    summary = result.summary()
    odds_radio = get_odds_radio(result)
    return result, summary, odds_radio


def ordinal_logit_regression_formula(data, formula):
    model = OrderedModel.from_formula(formula=formula, data=data)
    result = model.fit(method="bfgs")
    summary = result.summary()
    odds_radio = get_odds_radio(result)
    return result, summary, odds_radio


def ols_regression(x, y, intercept=True):
    if intercept:
        x = sm.add_constant(x)  # add constant if need intercept
    # run regression
    model = sm.OLS(y, x)
    result = model.fit()
    summary = result.summary()
    return result, summary


def ols_regression_formula(data, formula):
    # run regression
    model = sm.OLS.from_formula(formula=formula, data=data)
    result = model.fit()
    summary = result.summary()
    return result, summary


def add_interaction(x, *interactions):
    if len(interactions) <= 1:
        return
    if len(interactions) == 2:
        x[interactions[0] + " # " + interactions[1]] = (
            x[interactions[0]] * x[interactions[1]]
        )
    if len(interactions) == 3:
        x[interactions[0] + " # " + interactions[1]] = (
            x[interactions[0]] * x[interactions[1]]
        )
        x[interactions[0] + " # " + interactions[2]] = (
            x[interactions[0]] * x[interactions[2]]
        )
        x[interactions[1] + " # " + interactions[2]] = (
            x[interactions[1]] * x[interactions[2]]
        )
        x[interactions[0] + " # " + interactions[1] + " # " + interactions[2]] = (
            x[interactions[0]] * x[interactions[1]] * x[interactions[2]]
        )

import statsmodels.api as sm
import numpy as np
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


def get_odds_radio(result):
    # get Odds Ratio
    conf = result.conf_int()
    conf['Odds Ratio'] = result.params
    conf.columns = ['5%', '95%', 'Odds Ratio']
    return np.exp(conf)

def ordinal_logit_regression(x, y, intercept=True):
    model = OrderedModel(y, x, distr='logit')
    result = model.fit(method='bfgs')
    summary = result.summary()
    odds_radio = get_odds_radio(result)
    return result, summary, odds_radio

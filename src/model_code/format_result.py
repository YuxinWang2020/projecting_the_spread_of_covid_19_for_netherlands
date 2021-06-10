import pandas as pd
from statsmodels.iolib.summary2 import summary_col
from functools import reduce

def summary_to_stata_format(summary):
    summary = pd.DataFrame(summary.tables[1].data)
    summary = summary.set_axis(summary.iloc[0].astype(str), axis='columns').set_index('').iloc[1:, [0, 1, 3]]
    for i in range(len(summary)):
        if float(summary.iloc[i, 2]) < 0.01:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '***'
        elif float(summary.iloc[i, 2]) < 0.05:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '**'
        elif float(summary.iloc[i, 2]) < 0.1:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip() + '*'
        else:
            summary.iloc[i, 0] = summary.iloc[i, 0].lstrip()
        summary.iloc[i, 1] = '(' + summary.iloc[i, 1].lstrip() + ')'
    summary = summary.drop(columns='P>|z|').stack()
    summary = summary.reset_index(level=1,drop=True)
    return summary

def sm_results_format(results, model_names):
    return summary_col(results, float_format='%.3f', model_names=model_names, stars=True,
                       info_dict={'': lambda x: '',
                                  '': lambda x: '',
                                  'Observation': lambda x: str(int(x.nobs)),
                                  'R-squared': lambda x: f'{x.prsquared:.5f}',
                                  }
                       )

def odds_radio_format(odds_radios, model_names):
    return pd.concat([odds_radio['Odds Ratio'] for odds_radio in odds_radios], axis='columns', join='outer').set_axis(model_names, axis='columns')



import pandas as pd
import pytask

from src.config import BLD
from src.config import SRC


@pytask.mark.depends_on(SRC / "original_data" / "OxCGRT_latest.csv")
@pytask.mark.produces(BLD / "data" / "policy_stringency.pickle")
def task_OxCGRT_policy_stringency(depends_on, produces):
    # policy stringency index
    policy_data = pd.read_csv(depends_on).query("CountryCode=='NLD'")
    policy_data['Date'] = pd.to_datetime(policy_data['Date'], format='%Y%m%d', errors='ignore')
    policy_data = policy_data.set_index('Date')
    policy_stringency = policy_data[['C1_School closing', 'C2_Workplace closing', 'C6_Stay at home requirements']]
    policy_stringency.to_pickle(produces)

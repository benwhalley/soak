import pandas as pd
from pyirr import kappam_fleiss, read_data

# --- Example data (moderate agreement) ---
ratings = [
    [1, 1, 2],
    [1, 2, 2],
    [2, 2, 2],
    [2, 3, 2],
    [3, 3, 3],
    [3, 3, 3],
    [2, 2, 2],
    [1, 1, 1],
    [1, 2, 1],
    [2, 2, 2],
]


kappam_fleiss(ratings, detail=True)

pd.DataFrame(ratings).to_csv("~/dev/soak-package/tests/irrtest.csv")

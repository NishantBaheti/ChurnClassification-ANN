import pandas as pd
import numpy as np


def getData(completeFilePath, ext):
    """
    args -
    ----
    completeFilePath (string) = file path   
    ext(string) = extension csv,excel 

    returns -
    -------
    pd.DataFrame 
    """

    if ext.upper() == "CSV":
        return pd.read_csv(completeFilePath)

    elif ext.upper() == "EXCEL":
        return pd.read_excel(completeFilePath)

    else:
        return None

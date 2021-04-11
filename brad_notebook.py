#------------------------------------------------------
# Import necessary packages
#------------------------------------------------------

import numpy as np
import pandas as pd

#------------------------------------------------------
# Read in csv file
#------------------------------------------------------

nonvoters_df = pd.read_csv('nonvoters_data.csv')
print(nonvoters_df.shape)
print(nonvoters_df.columns)
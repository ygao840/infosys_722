import pandas as pd

input_file = 'LLCP2021.XPT'
output_file = 'LLCP2021.csv'

# Read the XPT file
df = pd.read_sas('LLCP2021.XPT')

# Save dataframe to CSV
df.to_csv(output_file, index=False)
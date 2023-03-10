import os
import pandas as pd

data = []

directory = r'C:\LumenDataScience\Datasets\Dataset\IRMAS_Validation_Data'

for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # check if the file is a text file
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as f:
            # read lines, remove whitespace and filter out empty lines
            lines = [line.strip() for line in f.readlines() if line.strip()]
            # join the instruments present in the file with a comma
            instruments = ", ".join(lines)
            data.append([filename, instruments])

# create a pandas dataframe with file name and instruments present in it
df = pd.DataFrame(data, columns=['File Name', 'Instruments'])

# display the dataframe
print(df)

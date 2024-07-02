# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

# Data Pre-Processing
transactions = []
for i in range(0, 7501):
    transactions.append([str(df.values[i,j]) for j in range(0, 20)])

# training the model
from apyori import apriori
rules = apriori(transactions=transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the result

# 1) Displaying first results coming directly from output of apriori function
results = list(rules)
# [print(result) for result in results]

# 2) Putting the result well organised into a pandas Dataframe
def inspect(lst):
    lhs = [tuple(el[2][0][0])[0] for el in lst]
    rhs = [tuple(el[2][0][1])[0] for el in lst]
    support = [el[1] for el in lst]
    confidence = [el[2][0][2] for el in lst]
    lift = [el[2][0][3] for el in lst]
    return list(zip(lhs, rhs, support, confidence, lift))

result_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'lift'])

# 3) Displaying the results non sorted
print(result_df)

# 4) Displaying the results sorted by descending lift
print('-----------------------------------------------------------------------------------')
print(result_df.nlargest(n=10, columns='lift'))
import pandas as pd
from apyori import apriori

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])
    
# Training the Eclat model on the dataset
rules = apriori(transactions = transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)

# Visualising the results
results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    support = [result[1] for result in results]
    return list(zip(lhs, rhs, support))


results_df = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# Displaying the results non sorted
print(results_df)

# Displaying the results sorted by descending lifts
print("\n\n")
print(results_df.sort_values(by='Support', ascending=False))
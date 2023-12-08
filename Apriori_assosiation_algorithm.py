import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Generate a synthetic dataset with continuous features
np.random.seed(42)
data = np.random.rand(100, 4) * 10  # 100 samples, 4 features
df = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'feature4'])

# Discretize the features into bins
num_bins = 5
for column in df.columns:
    df[column] = pd.cut(df[column], bins=num_bins, labels=[f'{column}_bin_{i}' for i in range(num_bins)])

# Create synthetic labels (categories)
df['label'] = np.random.choice(['A', 'B', 'C'], size=len(df))

# Split the dataset into training and testing sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Convert the training data into a list of transactions
transactions = df_train.values.tolist()

# Use Apriori algorithm to find frequent itemsets
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_freq_items = pd.DataFrame(te_ary, columns=te.columns_)

# Set minimum support threshold
min_support = 0.2
frequent_itemsets = apriori(df_freq_items, min_support=min_support, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the synthetic dataset, frequent itemsets, and association rules
print("Synthetic Dataset:")
print(df_train)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('RV_SAMPLES.csv')
df.drop(['URL', 'SKU', 'Notes', 'Status', 'Status.1', 'Product'], axis=1, inplace=True)
df['Pack'] = df['Pack'].astype(str)
df['jack'] = ''

# Remove rows where 'Year' cannot be converted to integer
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Drop rows with NaN values in 'Year'
df = df.dropna(subset=['Year'])

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

# Convert the 'Year' column to integers
df['Year'] = df['Year'].astype(int)

# Feature engineering for 'Pack' column
for i in range(len(df)):
    product_handle = df.loc[i, 'Pack']
    if product_handle and product_handle[0].isdigit():  # Check if the first character is a digit
        df.loc[i, 'jack'] = int(product_handle[0])
    else:
        df.loc[i, 'jack'] = np.nan  # Assign NaN for non-numeric first characters

df['jack'] = df['jack'].astype('Int64')  # Use Int64 dtype

df.drop(['Pack'], axis=1, inplace=True)

categorical_columns = ['RV Type', 'Manufacturer', 'Model Name', 'Trim', 'Leveling System']
df_encoded = pd.get_dummies(df, columns=categorical_columns)
x = df_encoded.drop(columns=['Product Handle'])
y = df_encoded['Product Handle']

# Ensure x and y do not have missing values
x = x.dropna()
y = y.dropna()

# Align the indices of x and y
x = x.loc[y.index]
y = y.loc[x.index]


# --------------------------------------------------------------------------------------------


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42) #Accuracy: 0.973724884080371 - this was the accuracy score when the number of decision trees were 40
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Optional: Test input for prediction
input_data = {
    'RV Type': ['Fifth Wheel'],  #
    'Manufacturer': ['Augusta'],
    'Model Name': [''],
    'Trim': [''],
    'Leveling System': [''],
    'jack': [6],  
    'Year': [2016]
}

test = pd.DataFrame(input_data)

categorical_columns2 = ['RV Type', 'Manufacturer', 'Model Name', 'Trim', 'Leveling System']
oneHotEncoding = pd.get_dummies(test, columns=categorical_columns2)
oneHotEncoding = oneHotEncoding.reindex(columns=x.columns, fill_value=0)

x_values = oneHotEncoding
prediction = model.predict(x_values)
print(prediction)
print('shape of y: ', y.shape)
print('shape of prediction: ', y_pred.shape)
print()
print('shape of x', x.shape)
print('shape of Testing model', x_test.shape)
print('shape of Training model', x_train.shape)

joblib.dump(model, 'random_forest_classifier.pkl')
joblib.dump(x.columns,'model_columns.pkl' )
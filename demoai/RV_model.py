import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('RV_SAMPLES.csv')

le_rv_type = LabelEncoder()
le_manufacturer = LabelEncoder()
le_model_name = LabelEncoder()
le_trim = LabelEncoder()
le_leveling_system = LabelEncoder()
le_year = LabelEncoder()

# Apply label encoding to categorical features
df['RV Type'] = le_rv_type.fit_transform(df['RV Type'])
df['Manufacturer'] = le_manufacturer.fit_transform(df['Manufacturer'])
df['Model Name'] = le_model_name.fit_transform(df['Model Name'])
df['Leveling System'] = le_leveling_system.fit_transform(df['Leveling System'])
df['Year'] = le_year.fit_transform(df['Year'])

# Initialize LabelEncoders for target variables
le_product = LabelEncoder()
le_pack = LabelEncoder()
le_url = LabelEncoder()
le_product_handle = LabelEncoder()
le_sku = LabelEncoder()

df['Product'] = le_product.fit_transform(df['Product'])
df['Pack'] = le_pack.fit_transform(df['Pack'])
df['Product Handle'] = le_product_handle.fit_transform(df['Product Handle'])


X = df[['RV Type', 'Manufacturer', 'Model Name', 'Leveling System', 'Year']]


# For Product
y_product = df['Product']
X_train, X_test, y_train, y_test = train_test_split(X, y_product, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score (Product):", accuracy_score(y_test, y_pred))

# For Pack
y_pack = df['Pack']
X_train, X_test, y_train, y_test = train_test_split(X, y_pack, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score (Pack):", accuracy_score(y_test, y_pred))


# For Product Handle
y_product_handle = df['Product Handle']
X_train, X_test, y_train, y_test = train_test_split(X, y_product_handle, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score (Product Handle):", accuracy_score(y_test, y_pred))


print('this is the shape of X - Train:',X_train.shape)
print('this is the shape of X:',X.shape)


y_test_product = le_product.inverse_transform(y_test)
y_pred_product = le_product.inverse_transform(y_pred)

result_df = pd.DataFrame({
    'Actual Product': y_test_product,
    'Predicted Product': y_pred_product
})

# Display the result DataFrame
print(result_df)


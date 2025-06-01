import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataframe = pd.read_csv('kc_house_data.csv')
print(dataframe.head(3))

print('*' * 50 + '\n' + ' ' * 15 + 'Data Exploration\n' + '*' * 50)
print('Number of instances:', dataframe.shape[0])
print('Number of Features:', dataframe.shape[1] - 1)
print('Summary: ', dataframe.describe())

print('*' * 50 + '\n' + ' ' * 15 + 'Data Preprocessing\n' + '*' * 50)
print('Null Values')
print(dataframe.isnull().sum())

print('Duplicate Rows')
print(dataframe.duplicated().sum())

dataframe.drop(['id', 'date'], axis=1, inplace=True)

features = ['sqft_lot', 'floors', 'bedrooms', 'bathrooms']
X = dataframe[features]
Y = dataframe['price']

#Test Train Split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
dump(scaler, 'scaler.pkl')

#Model Training
model = LinearRegression()
model.fit(x_train, y_train)

print('*' * 50 + '\n' + ' ' * 15 + 'Model Evaluation\n' + '*' * 50)
y_pred = model.predict(x_test)


print('Mean Square Error (Test Set):', mean_squared_error(y_test, y_pred))
dump(model, 'reg_model.joblib')

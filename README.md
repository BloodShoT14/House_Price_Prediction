# Multiple Linear Regression - Housing Case Study

This project focuses on building a multiple linear regression model to predict housing prices based on various features. The dataset used in this project is called "Housing.csv".

## Project Overview

The goal of this project is to analyze the housing data and develop a multiple linear regression model to predict housing prices. The dataset contains various features such as area, number of bedrooms and bathrooms, parking space, etc., which will be used to train the regression model.

## Project Steps

### Step 1: Reading and Understanding the Data

```python
import pandas as pd

# Read the housing dataset from the CSV file
data = pd.read_csv('Housing.csv')

# Explore the dataset
print(data.head())
print(data.shape)
print(data.describe())
```

### Step 2: Visualizing the Data

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Use pairplots to visualize relationships between variables
sns.pairplot(data)
plt.show()

# Use boxplots to identify patterns or correlations
sns.boxplot(x='area', y='price', data=data)
plt.show()
```

### Step 3: Data Preparation

```python
# Map categorical variables to numerical values
data['furnishingstatus'] = data['furnishingstatus'].map({'furnished': 1, 'unfurnished': 0})

# Create dummy variables for the "furnishingstatus" feature
data = pd.get_dummies(data, columns=['furnishingstatus'], drop_first=True)

# Remove unnecessary variables from the dataset
data = data.drop('location', axis=1)
```

### Step 4: Splitting the Data into Training and Testing Sets

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets using a 70:30 ratio
X = data.drop('price', axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

### Step 5: Building a Linear Model

```python
import statsmodels.api as sm

# Add a constant to the training data
X_train = sm.add_constant(X_train)

# Create a linear regression model using the training data
model = sm.OLS(y_train, X_train)

# Print the summary of the linear regression model
print(model.fit().summary())
```

### Step 6: Feature Selection and Model Building

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE

# Perform feature selection using variance inflation factor (VIF)
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Update the model by removing highly correlated or insignificant variables
X_train = X_train.drop(['feature1', 'feature2'], axis=1)

# Perform recursive feature elimination (RFE)
selector = RFE(model, n_features_to_select=5)
selector = selector.fit(X_train, y_train)

# Evaluate the model and calculate the VIFs for the final selected features
X_train_selected = X_train[X_train.columns[selector.support_]]
model_selected = sm.OLS(y_train, X_train_selected)
print(model_selected.fit().summary())
```

### Step 7: Residual Analysis

```python
# Analyze the residuals of the model on the training data
residuals = model_selected.fit().resid
# Perform residual

 analysis here
```

### Step 8: Making Predictions Using the Final Model

```python
from sklearn.preprocessing import StandardScaler

# Apply scaling on the test dataset
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Make predictions using the final model on the test data
X_test_selected = X_test_scaled[:, selector.support_]
y_pred = model_selected.fit().predict(sm.add_constant(X_test_selected))
```

### Step 9: Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score

# Compare the actual and predicted values using scatter plots
# Perform scatter plot here

# Evaluate the performance of the model using metrics such as RMSE and R-squared
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print('RMSE:', rmse)
print('R-squared:', r2)
```

## Prerequisites

- Python 3.x
- Required libraries: NumPy, Pandas, Matplotlib, Seaborn, StatsModels, Scikit-learn

## Conclusion

This project demonstrates the process of building a multiple linear regression model for predicting housing prices. The provided code and steps help understand the data, preprocess it, build the model, and evaluate its performance. Feel free to modify and experiment with the code to enhance the model or apply it to different datasets.

For detailed implementations and explanations, please refer to the project code.

## License

This project is licensed under the [MIT License](LICENSE).
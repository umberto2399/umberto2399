import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load and Inspect Data
data = pd.read_csv('telco.csv')

# Print the exact column names for verification
print("Columns in the dataset:")
print(data.columns)

# Identify categorical and numerical features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Drop columns that can cause data leakage
leakage_columns = ['Customer ID', 'Churn Label', 'Churn Score', 'Customer Status', 
                   'Churn Category', 'Churn Reason']

# Define target column and features
target = 'Churn Label'
features = [col for col in data.columns if col not in leakage_columns + [target]]

# Verify the features being used
print("Features selected for training:")
print(features)

# Ensure only valid features are used in the ColumnTransformer
numerical_features = [col for col in numerical_features if col in features]
categorical_features = [col for col in categorical_features if col in features]

print("Numerical features:")
print(numerical_features)

# Function to remove outliers using the IQR method
def remove_outliers_iqr(X, y):
    X_numerical = X[numerical_features]
    Q1 = X_numerical.quantile(0.25)
    Q3 = X_numerical.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X_numerical < (Q1 - 1.5 * IQR)) | (X_numerical > (Q3 + 1.5 * IQR))).any(axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    return X_clean, y_clean

print("Categorical features:")
print(categorical_features)

# Step 2: Split the dataset into training and testing sets
X = data[features]
y = data[target].apply(lambda x: 1 if x == 'Yes' else 0)  # Encode target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove outliers from the training set using the IQR method
X_train, y_train = remove_outliers_iqr(X_train, y_train)

# Step 3: Data Preparation Pipeline
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Encode categorical features
])

# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Step 4: Model Training and Hyperparameter Tuning
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the hyperparameters grid to search
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score from GridSearchCV
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_}")

# Use the best estimator from grid search to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Step 5: Save the Best Model
joblib.dump(best_model, 'churn_model_pipeline.pkl')

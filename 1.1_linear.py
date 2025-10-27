import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

FILE_NAME = ''     #salary
TARGET_COLUMN = ''   #salary

try:
    # Load and prepare data
    data = pd.read_csv(FILE_NAME)
    print(data.head())
    X, y = data.drop(TARGET_COLUMN, axis=1), data[TARGET_COLUMN]
    print(f"Data: '{FILE_NAME}', Target: '{TARGET_COLUMN}'")

    # Preprocessing
    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)])

    # Train model
    model = Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nR²: {r2_score(y_test, y_pred):.4f}, MSE: {mean_squared_error(y_test, y_pred):.4f}")

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
    plt.title('Actual vs Predicted'); plt.xlabel('Actual'); plt.ylabel('Predicted')
    plt.grid(True); plt.show()

    # Get prediction
    sample_data = {col: [float(val) if col in num_cols else val] for col in X.columns for val in [input(f"Enter '{col}': ")]}
    prediction = model.predict(pd.DataFrame(sample_data))[0]
    print(f"Prediction: {prediction:.2f}")

    # Plot 2: Train/Test Split
    if len(num_cols) == 1 and len(cat_cols) == 0:
        feature = num_cols[0]
        plt.figure(figsize=(10, 7))
        plt.scatter(X_train[feature], y_train, c='blue', label='Train', alpha=0.6)
        plt.scatter(X_test[feature], y_test, c='green', label='Test', alpha=0.6)
        
        X_range = pd.DataFrame({feature: np.linspace(X[feature].min(), X[feature].max(), 100)})
        plt.plot(X_range[feature], model.predict(X_range), 'red', label='Regression', lw=2)
        plt.scatter(sample_data[feature], prediction, c='black', marker='X', s=200, label=f'Pred: {prediction:.2f}')
        
        plt.title('Regression Line'); plt.xlabel(feature); plt.ylabel(TARGET_COLUMN)
        plt.legend(); plt.grid(True); plt.show()

except FileNotFoundError:
    print(f"File '{FILE_NAME}' not found")
except KeyError:
    print(f"Target '{TARGET_COLUMN}' not found")
except Exception as e:
    print(f"Error: {e}")
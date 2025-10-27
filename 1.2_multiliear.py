import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

FILE_NAME = '' # <--- EDIT THIS startup
TARGET_COLUMN = '' # <--- EDIT THIS profit

try:
    # --- 1. Load Data ---
    data = pd.read_csv(FILE_NAME)
    print(data.head())
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]
    print(f"Loaded '{FILE_NAME}'. Target: '{TARGET_COLUMN}'")

    # --- 2. Define Universal Preprocessing ---
    # This finds all number columns
    numeric_features = X.select_dtypes(include=np.number).columns
    # This finds all text/category columns
    categorical_features = X.select_dtypes(exclude=np.number).columns
    
    print(f"--- Found {len(numeric_features)} numeric features.")
    print(f"--- Found {len(categorical_features)} categorical features.")

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    # This processor applies the correct steps to the correct columns
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

    # --- 3. Create and Train Model Pipeline ---
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LinearRegression())])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    model.fit(X_train, y_train)

    # --- 4. Evaluate Model ---
    print("\n" + "="*20 + " Model Evaluation " + "="*20)
    y_pred = model.predict(X_test)
    print(f"R-squared (R²): {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")

    # --- 5. Plot Graph: Actual vs. Predicted ---
    # This is the most important graph for multiple regression.
    # A good model will have dots close to the red diagonal line.
    print("\nDisplaying Actual vs. Predicted plot...")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', c='red', lw=2)
    plt.title('Graph: Actual vs. Predicted Values')
    plt.xlabel('Actual Values'); plt.ylabel('Predicted Values')
    plt.grid(True); plt.show()

    # --- 6. User-Input Predictor ---
    print("\n" + "="*20 + " 🚀 Sample Predictor " + "="*20)
    sample_data = {}
    for col in X.columns:
        val = input(f"  Enter value for '{col}': ")
        sample_data[col] = [float(val) if col in numeric_features else val]
    
    sample_df = pd.DataFrame(sample_data)
    prediction = model.predict(sample_df)
    print(f"\n--- Prediction for {sample_data} ---> {prediction[0]:.2f}")

except FileNotFoundError:
    print(f"Error: File '{FILE_NAME}' not found.")
except KeyError:
    print(f"Error: Target '{TARGET_COLUMN}' not in data. Columns: {data.columns.tolist()}")
except Exception as e:
    print(f"An error occurred: {e}")

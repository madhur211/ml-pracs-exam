import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

if __name__ == "__main__":
    FILE_NAME, TARGET_COLUMN = '', ''  # <--- EDIT THESE networka ads/Purchased
    
    try:
        # Load data
        data = pd.read_csv(FILE_NAME)
        print(data.head())
        X, y = data.drop(TARGET_COLUMN, axis=1), data[TARGET_COLUMN]
        print(f"Loaded '{FILE_NAME}'. Target: '{TARGET_COLUMN}'")

        # Preprocessing pipeline
        numeric_features = X.select_dtypes(include=np.number).columns
        categorical_features = X.select_dtypes(exclude=np.number).columns
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])

        # Model pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', LogisticRegression())])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        model.fit(X_train, y_train)

        # Evaluation
        y_pred = model.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title('Graph 1: Confusion Matrix')
        plt.ylabel('Actual'); plt.xlabel('Predicted')
        plt.show()

        # Prediction
        sample_data = {col: [float(val) if col in numeric_features else val] 
                      for col in X.columns 
                      for val in [input(f"  Enter value for '{col}': ")]}
        
        prediction = model.predict(pd.DataFrame(sample_data))[0]
        print(f"\n--- Prediction for {sample_data} ---> {prediction}")

        # Scatter plot (if 2 numeric features)
        if len(numeric_features) == 2:
            col1, col2 = numeric_features[0], numeric_features[1]
            test_df = X_test.copy()
            test_df['Actual'] = y_test
            
            plt.figure(figsize=(10, 7))
            colors = ['red', 'green']
            for i, label in enumerate([0, 1]):
                plt.scatter(test_df[test_df['Actual'] == label][col1], 
                           test_df[test_df['Actual'] == label][col2], 
                           c=colors[i], label=f'Actual: {label}', alpha=0.6)
            
            pred_color = colors[prediction]
            plt.scatter(sample_data[col1], sample_data[col2], c=pred_color, 
                       label=f'Your Prediction: {prediction}', marker='X', s=200, edgecolor='black')
            
            plt.title('Graph 2: Test Set Results with Your Prediction')
            plt.xlabel(col1); plt.ylabel(col2)
            plt.legend(); plt.grid(True); plt.show()
        else:
            print(f"(Skipped scatter plot: Requires 2 numeric features, found {len(numeric_features)})")

    except FileNotFoundError:
        print(f"Error: File '{FILE_NAME}' not found.")
    except KeyError:
        print(f"Error: Target '{TARGET_COLUMN}' not in data. Columns: {data.columns.tolist()}")
    except Exception as e:
        print(f"An error occurred: {e}")
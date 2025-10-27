import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree # <-- Added for plotting trees

FILE_NAME = '3_bbagging_car_evaluation.csv' # <--- EDIT THIS
TARGET_COLUMN = 'class' # <--- EDIT THIS

try:
    # --- 1. Load Data ---
    data = pd.read_csv(FILE_NAME)
    print(data.head())

    print(f"Loaded '{FILE_NAME}'. Target: '{TARGET_COLUMN}'")

    # --- 2. Simple Preprocessing (LabelEncode all columns) ---
    # This approach is simple, like your PDF, and works best on
    # datasets that are mostly categorical.
    df_encoded = data.copy()
    encoders = {} # To store encoders for class names

    print("Encoding all columns...")
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        if col == TARGET_COLUMN:
            # Save the encoder for the target to get class names later
            encoders[col] = le

    # --- 3. Split Data ---
    X = df_encoded.drop(TARGET_COLUMN, axis=1)
    y = df_encoded[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- 4. Create and Train Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Training Random Forest model...")
    model.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    print("\n" + "="*20 + " Model Evaluation " + "="*20)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # --- 6. Plot Graph 1: Confusion Matrix ---
    print("\nDisplaying Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Graph 1: Confusion Matrix'); plt.ylabel('Actual'); plt.xlabel('Predicted'); plt.show()

    # --- 7. Plot Graph 2: Feature Importance ---
    print("Displaying Feature Importance plot...")
    importances = model.feature_importances_
    feature_names = X.columns
    
    feature_importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(x=feature_importance_series.values, y=feature_importance_series.index)
    plt.title('Graph 2: Feature Importance')
    plt.xlabel('Importance'); plt.ylabel('Feature'); plt.show()

    # --- 8. Plot Graph 3: Decision Trees (like your PDF) ---
    print("Displaying the first 3 Decision Trees...")
    
    # Get the class names back from the LabelEncoder
    try:
        class_names = encoders[TARGET_COLUMN].classes_.astype(str)
    except:
        class_names = np.unique(y).astype(str)

    # Loop to plot the first 3 trees
    for i in range(3):
        plt.figure(figsize=(20, 10)) # Make the figure large
        plot_tree(model.estimators_[i],  # model.estimators_ holds all the trees
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  max_depth=3, # Limit depth for readability
                  fontsize=10)
        plt.title(f'Decision Tree {i+1} from Random Forest (Max Depth: 3)')
        plt.show()

except FileNotFoundError:
    print(f"Error: File '{FILE_NAME}' not found.")
except KeyError:
    print(f"Error: Target '{TARGET_COLUMN}' not in data. Columns: {data.columns.tolist()}")
except Exception as e:
    print(f"An error occurred: {e}")
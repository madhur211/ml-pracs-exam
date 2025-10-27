import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from xgboost import XGBClassifier, plot_tree as xgb_plot_tree
import matplotlib.pyplot as plt

# 1. Load and prepare the data
df = pd.read_csv('3_boostingXgboost.csv')
X = df.drop(['Sample code number', 'Class'], axis=1)
y = df['Class'].map({2: 0, 4: 1})  # Map classes 2->0 and 4->1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# 3. Train Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 4. Evaluate both models
y_pred_xgb = xgb_model.predict(X_test)
y_pred_dt = dt_model.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt) * 100:.2f}%")

# 5. Plot XGBoost first tree
fig, ax = plt.subplots(figsize=(15, 10))
xgb_plot_tree(xgb_model, tree_idx=0, ax=ax)
plt.title("XGBoost - First Tree")
plt.savefig('xgboost_tree_0.pdf', bbox_inches='tight')
plt.close()
print("Saved XGBoost tree plot as 'xgboost_tree_0.pdf'")

# 6. Plot Decision Tree
fig, ax = plt.subplots(figsize=(15, 10)) 
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['Tumor (non-cancerous)', 'Tumor (cancerous)'], ax=ax)
plt.title("Decision Tree")
plt.savefig('decision_tree.pdf', bbox_inches='tight')
plt.close()
print("Saved Decision Tree plot as 'decision_tree.pdf'")

# 7. User input for prediction with XGBoost
print("\n--- Real-Time Prediction with XGBoost ---")
feature_names = X.columns.tolist()
user_input_dict = {}
for feature in feature_names:
    while True:
        try:
            value = float(input(f"Enter value for '{feature}' [1-10]: "))
            if 1 <= value <= 10:
                user_input_dict[feature] = value
                break
            else:
                print("Value must be between 1 and 10. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

new_data = pd.DataFrame([user_input_dict])
pred_class = xgb_model.predict(new_data)[0]
pred_proba = xgb_model.predict_proba(new_data)[0]
class_labels = {0: 'Tumor (non-cancerous)', 1: 'Tumor (cancerous)'}
print("\n--- Prediction Results ---")
print(f"Predicted class: {class_labels[pred_class]}")
print(f"Confidence {class_labels[0]}: {pred_proba[0] * 100:.2f}%")
print(f"Confidence {class_labels[1]}: {pred_proba[1] * 100:.2f}%")
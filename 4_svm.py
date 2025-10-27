import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from matplotlib.colors import ListedColormap

FILE_NAME = '2_logistic_Social_Network_Ads.csv'   #social
TARGET_COLUMN = 'Purchased'  #purchased

try:
    # Load data
    data = pd.read_csv(FILE_NAME)
    print(data.head())
    X, y = data.drop(TARGET_COLUMN, axis=1), data[TARGET_COLUMN]
    
    if len(X.columns) != 2:
        print(f"Need 2 features, found {len(X.columns)}")
        exit()
        
    f1, f2 = X.columns[0], X.columns[1]

    # Scale & split
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

    # Train model
    model = SVC(kernel='rbf', random_state=0)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Get prediction
    val1, val2 = float(input(f"Enter '{f1}': ")), float(input(f"Enter '{f2}': "))
    input_scaled = StandardScaler().fit(X).transform([[val1, val2]])
    prediction = model.predict(input_scaled)[0]
    print(f"Prediction: {prediction}")

    # Plot function
    def plot_boundary(X_set, y_set, title, pred_point=None):
        plt.figure(figsize=(9, 7))
        X1, X2 = X_set[:, 0], X_set[:, 1]
        xx1, xx2 = np.meshgrid(np.arange(X1.min()-1, X1.max()+1, 0.01),
                               np.arange(X2.min()-1, X2.max()+1, 0.01))
        
        Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=ListedColormap(('red', 'blue')))
        
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X1[y_set == j], X2[y_set == j], c=[['red', 'blue'][i]], 
                       label=j, s=25, edgecolor='k')
        
        if pred_point is not None:
            plt.scatter(pred_point[0][0], pred_point[0][1], c='black', 
                       marker='X', s=200, label=f'Prediction: {prediction}')

        plt.title(title)
        plt.xlabel(f1); plt.ylabel(f2)
        plt.legend(); plt.show()

    # Create plots
    plot_boundary(X_train, y_train.values, 'SVM (Training)')
    plot_boundary(X_test, y_test.values, 'SVM (Test)', input_scaled)

except FileNotFoundError:
    print(f"File '{FILE_NAME}' not found")
except KeyError:
    print(f"Target '{TARGET_COLUMN}' not found")
except Exception as e:
    print(f"Error: {e}")
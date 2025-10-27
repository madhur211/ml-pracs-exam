import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. Load Data ---
iris = load_iris()
X, y = iris.data, iris.target

# --- 2. Scale & Run PCA ---
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --- 3. Create DataFrame for Plotting ---
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Species'] = pd.Series(y).map({i: name for i, name in enumerate(iris.target_names)})

# --- 4. Print Results & Plot ---
print(f"Total Variance Explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
print("Explained variance ratio:", pca.explained_variance_ratio_)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Species', data=pca_df, s=70)
plt.title('PCA of Iris Dataset')
plt.grid(True); plt.show()

# x='PC1': Use the first component for the x-axis
# y='PC2': Use the second component for the y-axis
# hue='Species': Color the dots based on the 'Species' column












#This code loads the famous Iris dataset, uses Principal Component Analysis (PCA) to squish its four features (sepal length, sepal width, petal length, petal width) down to just two "main" features (PC1 and PC2), and then draws a 2D graph to see if the three flower species are clearly separated.
#x fetures y target
#StandardScaler(): This is a crucial step. It rescales all features so they have a mean of 0 and a standard deviation of 1. This prevents features with large values (like sepal length) from dominating PCA.
 #PCA(n_components=2): This tells PCA "I want to reduce all my features down to the best 2 components."
#pca.fit_transform(X_scaled): This is where the magic happens. It calculates the 2 new components (PC1 and PC2) from the scaled data. X_pca now holds the new data, which has only two columns.

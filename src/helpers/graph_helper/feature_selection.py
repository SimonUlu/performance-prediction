from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=500, n_informative=5, noise=0.2, random_state=42)

# Applying PCA
pca = PCA(n_components=2)  # Reducing to 2 dimensions for visualization
X_pca = pca.fit_transform(X)

# Plotting the principal components
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title('PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from google.colab import drive
from sklearn.decomposition import PCA

X = np.load('/content/drive/My Drive/X.npy')
y = np.load('/content/drive/My Drive/y.npy')

scaler = StandardScaler()

X_flatten = X.reshape(X.shape[0],-1)

X_standardized_flatten = scaler.fit_transform(X_flatten)
pca = PCA(n_components=300)
pca.fit(X_standardized_flatten)
X_pca = pca.transform(X_standardized_flatten)
X_train, X_test, y_train, y_test = train_test_split(X_pca,y,test_size=0.2, random_state=1)

cluster = KMeans(n_clusters=30, random_state=0, n_jobs=1)
model = cluster.fit(X_train)
predictions = model.predict(X_test)
print(classification_report(y_test,predictions))

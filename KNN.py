import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from google.colab import drive
from sklearn.decomposition import PCA
def classification_report_with_accuracy_score(y_true, y_pred):

    print(classification_report(y_true, y_pred))# print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

drive.mount('/content/drive')
X = np.load('/content/drive/My Drive/X.npy')
y = np.load('/content/drive/My Drive/y.npy')
scaler = StandardScaler()

X_flatten = X.reshape(X.shape[0],-1)

X_standardized_flatten = scaler.fit_transform(X_flatten)
pca = PCA(n_components=1000)
pca.fit(X_standardized_flatten)
X_pca = pca.transform(X_standardized_flatten)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
nested_score = cross_val_score(knn, X=X_pca, y=y, cv=3, scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)
print(nested_score.mean())


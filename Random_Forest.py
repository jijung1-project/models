import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
pca = PCA(n_components=50)
pca.fit(X_standardized_flatten)
X_pca = pca.transform(X_standardized_flatten)
random_forest = RandomForestClassifier(random_state=0, n_jobs=1)
nested_score = cross_val_score(random_forest, X=X_pca, y=y, cv=10, scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)
print(nested_score.mean())
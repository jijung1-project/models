import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, make_scorer
# from google.colab import drive
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
from matplotlib import rc
import matplotlib
import matplotlib.font_manager as fm

def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))# print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score

# drive.mount('/content/drive')
X = np.load('./X.npy')
y = np.load('./y.npy')

scaler = StandardScaler()

X_flatten = X.reshape(X.shape[0],-1)

X_standardized_flatten = scaler.fit_transform(X_flatten)
pca = PCA(n_components=50)
pca.fit(X_standardized_flatten)
X_pca = pca.transform(X_standardized_flatten)
logistic_regression = LogisticRegression(random_state=0, solver="saga")
nested_score = cross_val_score(logistic_regression, X=X_pca, y=y, cv=2, scoring=make_scorer(classification_report_with_accuracy_score))
print(nested_score)
print(nested_score.mean())

class_names = ['갈비구이', '갈비탕', '갈치구이', '감자채볶음', '계란찜', '김밥',
 '김치', '김치찌개', '깻잎장아찌' ,'꼬막찜' ,'된장찌개', '떡볶이',
 '만두' ,'물회', '삼계탕' ,'새우튀김' ,'소세지볶음' ,'수정과' ,'순대',
 '식혜' ,'약과' ,'욱회' ,'자장면' ,'족발' ,'찜닭' ,'콩자반' ,'피자',
 '한과', '해물찜' ,'후라이드치킨']
# class_names = ['a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a','a']

X_train, X_test, y_train, y_test = train_test_split(X_pca,y,test_size=0.2, random_state=1)
target_predicted = logistic_regression.fit(X_train, y_train).predict(X_test)

matrix = confusion_matrix(y_test, target_predicted)

dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
font_name = fm.FontProperties(fname=path).get_name()
plt.rc('font', family=font_name)
plt.title("Confusion Matrix"),
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

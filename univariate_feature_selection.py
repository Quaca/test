import pandas as pd
fajl = pd.read_csv("neki path.csv")

# izdvaja prvih 8 i zadnju kolonu kao y set
x = fajl.iloc[:,0:8]
y = fajl.iloc[:,-1]


# largest impact
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

best_features = SelectKBest(score_func = chi2, k=5)
model = best_features.fit(x,y)

dfscores = pd.DataFrame(model.scores_)
dfcolumns = pd.DataFrame(x.columns)

featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Features', 'Score']
print(featureScores.nlargest(5, 'Score')


#Feature importance 
from sklearn.ensamble import ExtraTreesClassifier
import matplotlib.pyplot as plt

model = ExtraTreesClassifier()
model.fit(x,y)

feature_importances = pd.Series(model.feature_importances_, index = x.columns)
feature_importances.nlargest(5).plot(kind='barh')
plt.show()


# Correlation
import seaborn as sns

correlations = fajl.corr()
top_cor_features = correlations.index
plt.figure(fig.size=(8,8))

heatmap_diabetes = sns.heatmap(fajl[top_cor_features].corr(), annot=True, cmap="RdY1Gn")
plt.show()

from sklearn.datasets import load_iris
iris = load_iris()

# store the feature matrix (X) and response vector (y) 
X = iris.data 
y = iris.target 
  
# store the feature and target names 
feature_names = iris.feature_names 
target_names = iris.target_names

# printing features and target names of our dataset 
print("Feature names:", feature_names) 
print("Target names:", target_names)
print("X value:", X)

print("Y value:", y)

# splitting X and y into training and testing sets 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# training the model on training set 
from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, y_train) 

# making predictions on the testing set 
y_pred = knn.predict(X_test) 
# comparing actual response values (y_test) with predicted response values (y_pred) 
from sklearn import metrics 
print("kNN model accuracy:", metrics.accuracy_score(y_test, y_pred))

# making prediction for out of sample data 
sample = [[3, 5, 4, 2], [2, 3, 5, 4]] 
preds = knn.predict(sample) 
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species) 
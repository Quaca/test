
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime
from sklearn.model_selection import train_test_split


df = pd.read_csv('attrition_train.csv', index_col=0)
print(df.head(1))

# Vidimo da se skup podataka sastoji od 35 atributa, sa ciljnom varijablom Attrition.
# Potrebno je prvo popuniti null vrijednosti u skupu podataka...

categorical_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Over18', 'BusinessTravel', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance', 'Attrition']
df = df.fillna(df.mode().iloc[0])

# Sve null vrijednosti u kategorickim kolonama popunjavamo sa najcescom vrijednoscu u toj koloni

numerical_cols = ['DailyRate', 'DistanceFromHome', 'EmployeeCount', 'EmployeeNumber', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'StandardHours','TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
df = df.fillna(df.mean().iloc[0])

# onehot encoding
to_onehot_encode_cols = ['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime', 'Over18']
df = pd.get_dummies(df, columns=to_onehot_encode_cols)

# zadnju kolonu birthdate pretvaramo u broj godina starosti:
df['Age'] = df['BirthDate'].map(lambda x: abs((datetime.now() - datetime.strptime(x, '%Y-%m-%d')).days) / 365)
df = df.drop('BirthDate', axis=1)

to_encode_cols = {
    'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
    'Education': ['1', '2', '3', 'Master', 'Doctor'],
    'EnvironmentSatisfaction': ['1', 'Medium', '3', '4'],
    'JobInvolvement': ['1', '2', 'High', '4'],
    'JobLevel': ['One', '2', '3', '4', 'Five'],
    'JobSatisfaction': ['1', '2', '3', 'Very High'],
    'PerformanceRating': ['Excellent', 'Outstanding'],
    'RelationshipSatisfaction': ['Low', '2', 'High', '4'],
    'StockOptionLevel': ['Zero', '1', '2', '3'],
    'WorkLifeBalance': ['Bad', 'Good', 'Better', 'Best'],
    'Attrition': ['No', 'Yes']
}

for feature, vals in to_encode_cols.items():
    df[feature] = df[feature].map(lambda x: vals.index(x))

# Potrebno je odvojiti ciljni atribut Attrition od ostatka skupa
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Nakon popunjenih nedostajucih vrijednosti mozemo preci na treneiranje modela

stablo = DecisionTreeClassifier()
random_forest = RandomForestClassifier()
gbm = GradientBoostingClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

stablo.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gbm.fit(X_train, y_train)

print(f'Tacnost stabla holdout metodom: {stablo.score(X_test, y_test)}')
print(f'Tacnost random foresta holdout metodom: {random_forest.score(X_test, y_test)}')
print(f'Tacnost boosted foresta holdout metodom: {gbm.score(X_test, y_test)}')

# # Vidimo da boosted forest daje najbolju tacnost ~ 85%

# Interval puzdanosti je procjena koja nam govori da ce uzorak koji je izabran iz populacije upasti u interval populacije tj. na osnovnu zadatih parametara populacije poput srednje vrijednosti ili varijance 
# nastojimo da odredimo u kojem intervalu srednja vrijednost uzorka lezi u uzorku populacije.
# P-vrijednost je vjerovatnoća koja određuje povrsinu desno od neke tačke kod normalne distribucije. 
# Pos predict value je vjerovatnoća da će slučajno odabrani uzorak biti ispravno pozitivno klasificiran.
# Prevalence je količina loših predikcija klasifikacije u cijeloj populaciji.
# Detection rate je omjer pozitivnih instanci koji su tačno klasificirane u odnosu na svim pozitivnim unutar populacije.
# Detection prevalence je broj prediktanih pozitivnih uzoraka  u odnosu na  cijelu populaciju.
# Balanced accuracy je srednja vrijednost tačnosti za svaku klasu klasifikacije pojedinačno.


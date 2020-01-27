import pandas as pd
import matplotlib.pyplot as plt
movies = pd.read_csv("path", sep=';', nrows=100)

one_hot = movies.genre.str.split("\s*,\s*, expand=True).stack().str.get_dummies().sum(level=0)
one_hot.head()

movies = movies.drop('genre', axis=1)
movies = movies.join(one_hot)


ratings = {}
for _,row in movies.iterrows():
	for col in COLUMNS:
		if row[col] == 1 and row['userid'] == 4579:
			if col not in ratings:
				ratings[col] = {
					"rating": row['rating'],
					"count":1
				}
			else:
				ratings[col]['rating'] += row['rating']
				ratings[col]['count'] += 1

GENRES = []
MEAN_RATING = []
print(ratings)
for x,y in ratings.items():
	GENRES.append(x)
	MEAN_RATING.append(float(y['rating'])/y['count'])
	plt.figure(figsize = 20,18)
	plt,figure(figsize = (20,10)
	plt.bar(GENRES, MEAN_RATING)
	plt.show()
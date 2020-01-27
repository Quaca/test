import pandas as pd
import numpy as np
import math
import ast
import matplotlib.pyplot as plt

movies = pd.read_csv("C:/Users/User/Desktop/backap dataset/movies_metadata.csv", sep=',', nrows=100)
movies = movies.drop('homepage', axis=1)
movies = movies.drop('video', axis=1)
movies = movies.drop('overview', axis=1)
movies = movies.drop('tagline', axis=1)
movies = movies.drop('status', axis=1)
movies = movies.drop('poster_path', axis=1)

movies['belongs_to_collection'].replace(np.nan,0, inplace=True)

movies.isnull().sum()

movies.describe()

to_replace = {
}
for x in movies["belongs_to_collection"]:
    if x is not 0:
        y = ast.literal_eval(x)
#         print(y.get("id"))
        to_replace.update(
            {
                str(x): int(y.get("id"))
            }
        )
                

movies["belongs_to_collection"].replace(
    to_replace,
    inplace=True
)

movies['adult'] = movies['adult'].replace(False,int(0))
movies['adult'] = movies['adult'].replace(True,int(1))

import json
import math
import numpy as np

for _, row in movies.iterrows():
    collection = row['belongs_to_collection']
    print(collection)
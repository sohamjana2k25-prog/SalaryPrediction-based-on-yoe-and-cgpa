import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


from lrclass import linear_regression

df = pd.read_csv('placement_with_rating.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=32)

model = linear_regression(0.0001, 50)
model.fit(X_train, Y_train)
with open('model.pkl', 'rb') as f:
    classifier = pickle.load(f)

import warnings

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

running = True
def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    #data was scraped by hand
    team_df = pd.read_excel("ESPN Data.xlsx")

    #convert team names to integer values
    le = LabelEncoder()
    label = le.fit_transform(team_df['Team'])
    team_df['Team (Numerical)'] = label

    figure, axis = plt.subplots(1, 3)

    #using the decision tree model
    decision_tree(team_df, axis)

    #using the naive bayes model
    naive_bayes(team_df, axis)

    #using KNN model
    knn(team_df, axis)

    # mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()

    plt.show()


def decision_tree(df, axis):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    d_tree = DecisionTreeClassifier().fit(X_train, y_train)
    d_tree_pred = d_tree.predict(X_test) 

    acc = accuracy_score(y_test, d_tree_pred)

    wins = 0
    losses = 0
    for i in d_tree_pred:
        if i == 1:
            losses += 1
        else:
            wins += 1
    title = ['Wins', 'Losses']
    record = [wins, losses]
    axis[0].bar(title, record)
    axis[0].set_title("Decision Tree")
    axis[0].set_xlabel("Model accuracy: " + str(round(acc, 2)))
    print("Wins: " + str(wins))
    print("Losses: " + str(losses))
    print("Decision tree model accuracy: " + str(round(acc, 2)) + "\n")
 

def naive_bayes(df, axis):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    naive = MultinomialNB().fit(X_train, y_train)
    naive_predict = naive.predict(X_test)

    acc = accuracy_score(y_test, naive_predict)

    wins = 0
    losses = 0
    for i in naive_predict:
        if i == 1:
            losses += 1
        else:
            wins += 1
    title = ['Wins', 'Losses']
    record = [wins, losses]
    axis[1].bar(title, record)
    axis[1].set_title("Naive Bayes")
    axis[1].set_xlabel("Model accuracy: " + str(round(acc, 2)))
    print("Wins: " + str(wins))
    print("Losses: " + str(losses))
    print("Naive Bayes accuracy: " + str(round(acc, 2)) + "\n")


def knn(df, axis):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    pipeline = Pipeline([('mms', MaxAbsScaler()), ('knn', KNeighborsClassifier())])

    """we used various parameters until we found the ones that fit best"""
    params = [{'knn__n_neighbors': [4, 5, 6, 7], 'knn__weights': ['uniform', 'distance'], 'knn__leaf_size': [3,4,5,6,7]}]
    grid_search = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    """testing to see which knn had the best accuracy
    best was leaf-size = 5, neighbors = 5, weight = uniform"""
    knn_test = KNeighborsClassifier(leaf_size=5, n_neighbors=4, weights='uniform')
    knn_test.fit(X_train, y_train)
    predict_knn = knn_test.predict(X_test)
    acc = accuracy_score(y_test, predict_knn)

    wins = 0
    losses = 0
    for i in predict_knn:
        if i == 1:
            losses += 1
        else:
            wins += 1
    title = ['Wins', 'Losses']
    record = [wins, losses]
    axis[2].bar(title, record)
    axis[2].set_title("KNN")
    axis[2].set_xlabel("Model accuracy: " + str(round(acc, 2)))
    print("Wins: " + str(wins))
    print("Losses: " + str(losses))
    print("KNN accuracy: " + str(round(acc, 2)))


if running:
    main()
    running = False

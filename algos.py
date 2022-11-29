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

def main():
    #data was scraped by hand
    team_df = pd.read_excel("ESPN Data.xlsx")

    #convert team names to integer values
    le = LabelEncoder()
    label = le.fit_transform(team_df['Team'])
    team_df['Team (Numerical)'] = label
    
    #using the decision tree model
    decision_tree(team_df)

    #using the naive bayes model
    naive_bayes(team_df)

    #using KNN model
    knn(team_df)

def decision_tree(df):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    d_tree = DecisionTreeClassifier().fit(X_train, y_train)
    d_tree_pred = d_tree.predict(X_test) 

    acc = accuracy_score(y_test, d_tree_pred)
    print("Decision tree model accuracy: " + str(round(acc, 2)))
 

def naive_bayes(df):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    naive = MultinomialNB().fit(X_train, y_train)
    naive_predict = naive.predict(X_test)

    acc = accuracy_score(y_test, naive_predict)
    print("Naive Bayes accuracy: " + str(round(acc, 2)))
    
def knn(df):
    features = ['W/L?', 'Team (Numerical)']
    X = df.loc[:, features]
    y = df['National Championship']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    pipeline = Pipeline([('mms', MaxAbsScaler()), ('knn', KNeighborsClassifier())])
    #we used various parameters until we found the ones that fit best
    params = [{'knn__n_neighbors': [4, 5, 6, 7], 'knn__weights': ['uniform', 'distance'], 'knn__leaf_size': [3,4,5,6,7]}]
    grid_search = GridSearchCV(pipeline, param_grid=params, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    
    #testing to see which knn had the best accuracy
    #best was leaf-size = 5, neighbors = 5, weight = uniform
    knn_test = KNeighborsClassifier(leaf_size=5, n_neighbors=4, weights='uniform')
    knn_test.fit(X_train, y_train)
    predict_knn = knn_test.predict(X_test)
    acc = accuracy_score(y_test, predict_knn)
    print("KNN accuracy: " + str(round(acc,2)))





if __name__ == '__main__':
    main()

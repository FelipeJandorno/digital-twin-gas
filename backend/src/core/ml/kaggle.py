import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings

warnings.filterwarnings('ignore')

data = 'C:\\Users\\marco\\OneDrive\\Área de Trabalho\\Hackathon\\digital-twin-gas\\backend\\data\\synthetic\\generated\\dataset.csv'
df = pd.read_csv(data)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import category_encoders as ce

TARGET_VARIABLE = "leak_label"
FEATURE_VECTOR = df.columns.drop(["leak_label", "leak_status", "timestamp", "hour", "day_of_week", "month"])

def explore_data():
    print(df.shape)

def main():

    y = df[TARGET_VARIABLE]
    X = df.drop(["leak_label", "leak_status", "timestamp", "hour", "day_of_week", "month"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    encoder = ce.OrdinalEncoder(cols=FEATURE_VECTOR)

    X_train = encoder.fit_transform(X_train)
    X_test = encoder.fit_transform(X_test)

    rfc = RandomForestClassifier(n_estimators=10,random_state=0)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    print("\n------------\nFEATURES SCORES\n------------")
    feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_scores)

    print("\n------------\nCONFUSION MATRIX\n------------")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

main()
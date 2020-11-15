import joblib
import sklearn as sk
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier

dataframe = pd.read_csv(
    "https://raw.githubusercontent.com/o-KezoW/machineLearning-project" +
    "/main/src/data/pima-indians-diabetes.csv", header=None
    )

X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]

normalize = preprocessing.MinMaxScaler()
normalized_X = normalize.fit_transform(X)

X_train, X_test, y_train, y_test = (
    model_selection.train_test_split(normalized_X, y, test_size=0.30,
                                     random_state=42)
    )

classifier = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10),
    random_state=1, max_iter=500
    )

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

if __name__ == "main":
    print(sk.metrics.confusion_matrix(y_test, y_pred))
    print(sk.metrics.classification_report(y_test, y_pred))

# Saving our model to disk
file_name = "../data/best_model.sav" if __name__ == "__main__" else "./data/best_model.sav"  # Path and file name
joblib.dump(classifier, file_name)

load_file = joblib.load(file_name)

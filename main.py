from NaiveBayes import ClassifierNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = [
    ["Machine-op-inspct", "Own-child", "Black", "Male", "United-States", "Private", "<=50K"],
    ["Farming-fishing", "Husband", "White", "Male", "United-States", "Private", "<=50K"],
    ["Protective-serv", "Husband", "White", "Male", "United-States", "Local-gov", ">50K"],
    ["Machine-op-inspct", "Husband", "Black", "Male", "United-States", "Private", ">50K"],
    ["Other-service", "Not-in-family", "White", "Male", "United-States", "Self-emp", "<=50K"],
    ["Prof-specialty", "Husband", "White", "Male", "United-States", "Private", ">50K"],
    ["Other-service", "Unmarried", "White", "Female", "United-States", "Self-emp", "<=50K"],
    ["Craft-repair", "Husband", "White", "Male", "United-States", "Self-emp-not-inc", "<=50K"],
    ["Machine-op-inspct", "Husband", "White", "Male", "United-States", "Private", ">50K"],
    ["Adm-clerical", "Husband", "White", "Male", "United-States", "Private", "<=50K"],
    ["Adm-clerical", "Not-in-family", "White", "Female", "United-States", "Private", "<=50K"],
    ["Machine-op-inspct", "Husband", "White", "Male", "United-States", "Federal-gov", ">50K"],
    ["Exec-managerial", "Husband", "White", "Male", "United-States", "Private", ">50K"],
    ["Other-service", "Own-child", "White", "Male", "United-States", "Self-emp", "<=50K"],
    ["Adm-clerical", "Wife", "White", "Female", "United-States", "Private", "<=50K"],
    ["Machine-op-inspct", "Unmarried", "White", "Female", "United-States", "Private", "<=50K"],
    ["Never-married", "Priv-house-serv", "Not-in-family", "White", "Guatemala", "Private", "<=50K"],
    ["Never-married", "Machine-op-inspct", "Not-in-family", "White", "United-States", "Private", "<=50K"],
    ["Never-married", "Craft-repair", "Own-child", "White", "United-States", "Private", "<=50K"],
    ["Married-civ-spouse", "Prof-specialty", "Husband", "White", "United-States", "Private", ">50K"],
    ["Married-civ-spouse", "Sales", "Husband", "White", "United-States", "Private", ">50K"],
    ["Married-civ-spouse", "Farming-fishing", "Husband", "White", "United-States", "Self-emp", "<=50K"],
    ["Married-civ-spouse", "Other-service", "Husband", "White", "United-States", "Private", "<=50K"],
    ["Never-married", "Farming-fishing", "Own-child", "White", "United-States", "Private", "<=50K"],
    ["Married-civ-spouse", "Prof-specialty", "Wife", "White", "United-States", "Self-emp-not-inc", ">50K"]
]
X = [row[:-1] for row in data]  # Extract features (all columns except the last one)
y = [row[-1] for row in data]    # Extract labels (last column)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = ClassifierNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(accuracy_score(y_test, y_pred))

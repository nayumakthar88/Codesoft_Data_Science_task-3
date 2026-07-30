import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

current_folder = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_folder, "Iris.csv")

df = pd.read_csv(csv_file)

print("First 5 Rows\n")
print(df.head())

print("\nDataset Information\n")
print(df.info())

print("\nMissing Values\n")
print(df.isnull().sum())

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x="species", data=df)
plt.title("Species Count")
plt.show()

df_plot = pd.read_csv(csv_file)

sns.pairplot(df_plot, hue="species")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(
    df.drop("species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

for feature in features:
    plt.figure(figsize=(6,4))
    sns.histplot(df_plot[feature], bins=20, kde=True)
    plt.title(feature)
    plt.show()

print("\nProject Completed Successfully!")
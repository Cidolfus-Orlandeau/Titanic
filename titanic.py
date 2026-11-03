import pandas as pd
df = pd.read_csv("train.csv")

print(df)
print(df.columns)
print(df.info())
print(df.groupby("Sex")["Survived"].mean())
print(df.groupby("Pclass")["Survived"].mean())
print(df.groupby(["Sex", "Pclass"])["Survived"].mean())
print(df.groupby(["Sex", "Pclass"]).size())

#생존율과 성별, 신분의 연관성이 짙다.

df["AgeGroup"] = pd.cut(df["Age"], bins = [0, 10, 20, 30, 40, 50, 60, 120], labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61+"])
print(df.groupby("AgeGroup")["Survived"].mean())

print(df.groupby("SibSp")["Survived"].mean())
print(df.groupby("Parch")["Survived"].mean())
print(df.groupby("Embarked")["Survived"].mean())

print(df.Fare.max())
print(df.Fare.min())

df["FareGroup"] = pd.cut(df["Fare"], bins = [0, 100, 200, 300, 400, 500, 600], labels = ["0-100", "101-200", "201-300", "301-400", "401-500", "501-600"])
print(df.groupby("FareGroup")["Survived"].mean())
print(df.groupby("FareGroup").size())

#0-100이 대부분

print(df.groupby("Cabin")["Survived"].mean())

df["Deck"] = df["Cabin"].str[0]
print(df.groupby("Deck")["Survived"].mean())
print(df.groupby("Deck")["Survived"].size())

print(df.groupby(["Deck", "Pclass"]).size())

#Cabin은 결측치가 너무 많음

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Deck"] = df["Deck"].fillna("Unknown")

print(df.head())

df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
print(df[["Sex", "Pclass", "Age", "Fare", "Deck", "FamilySize"]].head())

df = pd.get_dummies(df, columns=["Deck"], dtype=int)

print(df.head())

X = df.drop("Survived", axis = 1)
y = df["Survived"]

print(X.head())
print(y.head())

test_df = pd.read_csv("test.csv")
print(test_df.head())
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
test_df["Deck"] = test_df["Cabin"].str[0]
test_df["Deck"] = test_df["Deck"].fillna("Unknown")

print(test_df["Age"].isna().sum())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())
test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].median())
test_df = pd.get_dummies(test_df, columns=["Deck"], dtype = int)
test_df = test_df.reindex(columns=X.columns, fill_value=0)

print(test_df.head())
print(X.columns)
print(test_df.columns)

# Logistic Regression 머신러닝 모델 만들기

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

# Random Forest

#from sklearn.ensemble import RandomForestClassifier

#model = RandomForestClassifier(
#    n_estimators = 100,
#    random_state = 42
#)

#Decision Tree

#from sklearn.tree import DecisionTreeClassifier

#model = DecisionTreeClassifier(
#    random_state = 42
#)

#Gradient Boosting

#from sklearn.ensemble import GradientBoostingClassifier

#model = GradientBoostingClassifier(
#    n_estimators = 100,
#    learning_rate = 0.1,
#    max_depth = 3,
#    random_state = 42
#)

#name에서 title 뽑기

df["Title"] = df["Name"].str.split(",").str[1].str.split(".").str[0].str.strip()
print(df["Title"].head(10))

#str은 문자열을 나타냄

print(df["Title"].value_counts())

def change_title(title):
    if title in ["Mr", "Mrs", "Miss", "Master"]:
        return title
    else:
        return "Rare"

df["TitleGroup"] = df["Title"].apply(change_title)

print(df["TitleGroup"].value_counts())

print(df.groupby("TitleGroup")["Survived"].mean())

df = pd.get_dummies(df, columns = ["TitleGroup"], dtype = int)

print(df.head())

print(df.groupby("Embarked").size())
print(df.groupby("Embarked")["Survived"].mean())

print(df.groupby(["Embarked", "Pclass", "Sex"])["Survived"].mean())
print(df.groupby(["Embarked", "Pclass", "Sex"]).size())

#C: 1등, 3등시민이 많이 탐, 남성이 살짝 많음
#Q: 3등시민, 남녀동비율
#S: 123 중 3이 제일 많고 남성이 훨씬 많음

print(df.groupby(["Embarked", "AgeGroup"])["Survived"].mean())
print(df.groupby(["Embarked", "AgeGroup"]).size())

test_df["Title"] = test_df["Name"].str.split(",").str[1].str.split(".").str[0].str.strip()
test_df["TitleGroup"] = test_df["Title"].apply(change_title)
test_df = pd.get_dummies(test_df, columns = ["TitleGroup"], dtype = int)

X = df[
    [
        "Sex",
        "Pclass",
        "Age",
        "FamilySize",
        "TitleGroup_Master",
        "TitleGroup_Miss",
        "TitleGroup_Mr",
        "TitleGroup_Mrs",
        "TitleGroup_Rare"
    ]
]

y = df["Survived"]

test_passenger_id = test_df["PassengerId"]

test_df = test_df.reindex(columns = X.columns, fill_value = 0)

model.fit(X,y)
test_X = test_df[X.columns]
predictions = model.predict(test_X)
print(predictions)

submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Survived": predictions
})

submission.to_csv("submission.csv", index=False)

print(submission.head())

#Logistic Regression에 대해
#자신만의 계수를 만들어서 각각 계산한 후 0.5보다 높으면 생존

print(X.columns)
print(test_X.columns)

print(X.shape)
print(test_X.shape)

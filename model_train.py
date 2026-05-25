import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('social_media_impact_on_mental_health.csv')

print("Dataset Loaded Successfully!\n")

def classify_risk(score):

    if score <= 2:
        return "Low Risk"

    elif score <= 4:
        return "Medium Risk"

    else:
        return "High Risk"

data['Risk_Level'] = data['mental_health_effect'].apply(classify_risk)


X = data[[
    'usage_level',
    'anxiety',
    'low_self_esteem',
    'depression',
    'social_isolation',
    'social_connection',
    'positive_self_image'
]]

y = data['Risk_Level']


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Size:", len(X_train))
print("Testing Size:", len(X_test))


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X_train, y_train)

print("\nModel Training Completed!")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)


print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

with open('mental_health_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

url = "E:\Git_files\model\heart.csv"
Heart = pd.read_csv(url)
X = Heart.drop("target", axis=1)

y = Heart["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


#save the model to a joblib file
joblib.dump(model, 'model/Heart_model.joblib')

print("Model trained and saved successfully.")
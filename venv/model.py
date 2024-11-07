import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv(r'C:\Users\Prasanna\Documents\GitHub\CURABOT\datasets\diseasehackathon.csv') #Prajwal


# Selecting relevant features and the target
features = ['Symptom', 'Severity', 'Additional Symptoms', 'Temperature', 'Humidity', 'Weather Conditions']
target = 'Action Needed'

# Drop any rows with missing values in critical columns
data = data.dropna(subset=features + [target])

# Splitting the data into features and target
X = data[features]
y = data[target]

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the column transformer for preprocessing
# Categorical columns will be one-hot encoded, while numerical columns will be standardized
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Symptom', 'Severity', 'Additional Symptoms', 'Humidity', 'Weather Conditions']),
        ('num', StandardScaler(), ['Temperature'])
    ])

# Creating the pipeline with preprocessing and model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Training the model
pipeline.fit(X_train, y_train)

#Saving the model
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Making predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluating the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
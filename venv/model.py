import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
#data = pd.read_csv(r'C:\Users\tarun\Cura\datasets\diseasehackathon.csv')

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

# Making predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluating the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))




import numpy as np

# Define a function to get user input, make a prediction, and display health alert and remedy advice
def manual_test_with_advice(pipeline, data):
    # Get unique values for each categorical column from the dataset
    symptoms = data['Symptom'].unique()
    severities = data['Severity'].unique()
    additional_symptoms = data['Additional Symptoms'].unique()
    humidity_levels = data['Humidity'].unique()
    weather_conditions = data['Weather Conditions'].unique()
    locations = data['User Location'].unique()
    
    # Prompt the user for input from the dataset options
    print("Please select from the following options.")
    
    location = input(f"Choose a Location {locations}: ")
    symptom = input(f"Choose a Symptom {symptoms}: ")
    severity = input(f"Choose a Severity {severities}: ")
    additional_symptom = input(f"Choose Additional Symptom {additional_symptoms}: ")
    temperature = float(input("Enter Temperature (float): "))
    humidity = input(f"Choose Humidity {humidity_levels}: ")
    weather = input(f"Choose Weather Conditions {weather_conditions}: ")

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[symptom, severity, additional_symptom, temperature, humidity, weather]], 
                              columns=['Symptom', 'Severity', 'Additional Symptoms', 'Temperature', 'Humidity', 'Weather Conditions'])
    
    # Make a prediction
    action_needed = pipeline.predict(input_data)[0]
    
    # Retrieve the recommended remedy from the dataset for the chosen symptom and severity
    remedy = data[(data['Symptom'] == symptom) & (data['Severity'] == severity)]['Remedy Type'].values
    remedy_text = remedy[0] if len(remedy) > 0 else "No specific remedy available"
    
    # Display the action advice based on the predicted action needed
    if action_needed == 0:
        print(f"Here are some home remedies: {remedy_text}. If it persists longer, please consult a doctor.")
    elif action_needed == 1:
        print(f"Here are some home remedies: {remedy_text}. If the severity of these symptoms increases, please consult a doctor.")
    elif action_needed == 2:
        print("Please consult a doctor.")
    
    # Display the health alert for the chosen location
    location_alerts = data[data['User Location'] == location]['health_alert'].unique()
    if location_alerts.size > 0:
        print(f"Health Alert for {location}: {', '.join(location_alerts)}")
    else:
        print(f"No specific health alerts for {location}.")

# Run the manual test function
manual_test_with_advice(pipeline, data)


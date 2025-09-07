from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

# Load columns used in training (after one-hot encoding)
trained_columns = joblib.load('columns.pkl')

# List of categorical columns that were one-hot encoded
categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines', 
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection', 
                    'TechSupport','StreamingTV','StreamingMovies','Contract',
                    'PaperlessBilling','PaymentMethod']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    input_data = request.form.to_dict()
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Ensure correct dtypes for numeric columns if needed
    # Example: df['tenure'] = pd.to_numeric(df['tenure'])
    
    # Apply one-hot encoding like during training
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all trained columns exist in the input
    df_encoded = df_encoded.reindex(columns=trained_columns, fill_value=0)
    
    # Make prediction
    prediction = model.predict(df_encoded)[0]
    
    return render_template('index.html', prediction_text=f'Prediction: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
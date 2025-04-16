from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

model_folder = 'models'
models = {
    name: joblib.load(os.path.join(model_folder, f'{name}.pkl'))
    for name in ['LogisticRegression', 'DecisionTree', 'RandomForest', 'SVC', 'KNN']
}

scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(model_folder, 'label_encoder.pkl'))

feature_names = [
    {"key": "WBC", "name": "White Blood Cell Count (WBC)"},
    {"key": "LYMp", "name": "Lymphocyte Percentage (LYMp)"},
    {"key": "NEUTp", "name": "Neutrophil Percentage (NEUTp)"},
    {"key": "LYMn", "name": "Lymphocyte Count (Absolute) (LYMn)"},
    {"key": "NEUTn", "name": "Neutrophil Count (Absolute) (NEUTn)"},
    {"key": "RBC", "name": "Red Blood Cell Count (RBC)"},
    {"key": "HGB", "name": "Hemoglobin (HGB)"},
    {"key": "HCT", "name": "Hematocrit (HCT)"},
    {"key": "MCV", "name": "Mean Corpuscular Volume (MCV)"},
    {"key": "MCH", "name": "Mean Corpuscular Hemoglobin (MCH)"},
    {"key": "MCHC", "name": "Mean Corpuscular Hemoglobin Concentration (MCHC)"},
    {"key": "PLT", "name": "Platelet Count (PLT)"},
    {"key": "PDW", "name": "Platelet Distribution Width (PDW)"},
    {"key": "PCT", "name": "Procalcitonin (PCT)"}
]

project_info = {
    'description': 'This application uses machine learning to diagnose blood disorders based on Complete Blood Count (CBC) parameters. Multiple models are available for comparison.',
    'features_info': {
        'WBC': {
            'name': 'White Blood Cell Count',
            'description': 'Measures the number of white blood cells in your blood, which are crucial for fighting infections.',
            'normal_range': '4,000 - 11,000 cells/μL'
        },
        'LYMp': {
            'name': 'Lymphocyte Percentage',
            'description': 'Indicates the percentage of lymphocytes, a type of white blood cell, in your blood.',
            'normal_range': '20% - 40%'
        },
        'NEUTp': {
            'name': 'Neutrophil Percentage',
            'description': 'Represents the percentage of neutrophils, another type of white blood cell, in your blood.',
            'normal_range': '40% - 60%'
        },
        'LYMn': {
            'name': 'Lymphocyte Count (Absolute)',
            'description': 'Measures the absolute number of lymphocytes in your blood.',
            'normal_range': '1,000 - 4,800/μL'
        },
        'NEUTn': {
            'name': 'Neutrophil Count (Absolute)',
            'description': 'Measures the absolute number of neutrophils in your blood.',
            'normal_range': '2,500 - 7,000/μL'
        },
        'RBC': {
            'name': 'Red Blood Cell Count',
            'description': 'Determines the number of red blood cells, which carry oxygen throughout your body.',
            'normal_range': 'Female: 4.2-5.4 million/μL\nMale: 4.7-6.1 million/μL'
        },
        'HGB': {
            'name': 'Hemoglobin',
            'description': 'Measures the amount of hemoglobin, the oxygen-carrying protein in red blood cells.',
            'normal_range': 'Female: 12.3-15.3 g/dL\nMale: 14.0-17.5 g/dL'
        },
        'HCT': {
            'name': 'Hematocrit',
            'description': 'Indicates the proportion of your blood made up of red blood cells.',
            'normal_range': 'Female: 36%-44%\nMale: 41%-50%'
        },
        'MCV': {
            'name': 'Mean Corpuscular Volume',
            'description': 'Measures the average size of your red blood cells.',
            'normal_range': '80-100 fL'
        },
        'MCH': {
            'name': 'Mean Corpuscular Hemoglobin',
            'description': 'Calculates the average amount of hemoglobin per red blood cell.',
            'normal_range': '27-31 pg/cell'
        },
        'MCHC': {
            'name': 'Mean Corpuscular Hemoglobin Concentration',
            'description': 'Assesses the average concentration of hemoglobin in your red blood cells.',
            'normal_range': '32-36 g/dL'
        },
        'PLT': {
            'name': 'Platelet Count',
            'description': 'Measures the number of platelets, which are essential for blood clotting.',
            'normal_range': '150,000-400,000/μL'
        },
        'PDW': {
            'name': 'Platelet Distribution Width',
            'description': 'Indicates the variation in platelet size, which can reflect platelet activation.',
            'normal_range': '8.3%-56.6%'
        },
        'PCT': {
            'name': 'Procalcitonin',
            'description': 'A marker that can indicate bacterial infections and sepsis.',
            'normal_range': 'Normal: <0.1 μg/L\nLow risk: <0.5 μg/L\nPossible sepsis: 0.5-2 μg/L\nHigh risk: 2-10 μg/L\nSevere: >10 μg/L'
        }
    }
}

@app.route('/')
def home():
    return render_template('index.html', 
                         models=models.keys(), 
                         features=feature_names,
                         project_info=project_info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html', project_info=project_info)

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not request.form:
        return render_template('index.html', 
                             models=models.keys(), 
                             features=feature_names,
                             project_info=project_info)
    
    try:
        # Validate that all required fields are present
        missing_fields = [f["name"] for f in feature_names if f["name"] not in request.form or not request.form[f["name"]]]
        if missing_fields or 'model' not in request.form:
            raise ValueError("All fields are required. Please fill in all values.")
            
        input_data = [float(request.form[f["name"]]) for f in feature_names]
        model_name = request.form['model']
        
        # Validate ranges for input data
        if not all(-1000 <= x <= 1000 for x in input_data):  # reasonable range for blood parameters
            raise ValueError("Input values must be within reasonable range (-1000 to 1000)")
            
        model = models[model_name]
        scaled_input = scaler.transform([input_data])
        prediction = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)[0]
        confidence = round(max(probabilities) * 100, 2)
        result = label_encoder.inverse_transform(prediction)[0]
        
        return render_template('index.html', 
                             prediction=result,
                             confidence=confidence,
                             models=models.keys(), 
                             features=feature_names,
                             project_info=project_info,
                             input_values=dict(zip([f["name"] for f in feature_names], input_data)))
    except ValueError as ve:
        return render_template('index.html', 
                             prediction=f"Error: {str(ve)}", 
                             models=models.keys(), 
                             features=feature_names,
                             project_info=project_info)
    except Exception as e:
        return render_template('index.html', 
                             prediction=f"Error: An unexpected error occurred. Please check your input values.", 
                             models=models.keys(), 
                             features=feature_names,
                             project_info=project_info)

if __name__ == '__main__':
    app.run(debug=True)

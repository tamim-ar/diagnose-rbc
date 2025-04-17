# DiagnoseRBC - Blood Disorder Diagnosis System

A web-based application for diagnosing blood disorders using machine learning algorithms.

## Overview

DiagnoseRBC is a system that analyzes blood test parameters to help diagnose various blood disorders including:

- Healthy conditions
- Iron deficiency anemia
- Normocytic normochromic anemia
- Other microcytic anemia
- Leukemia
- Leukemia with thrombocytopenia

## Features

- Multiple ML models for blood disorder classification
- Interactive web interface for data input
- Detailed parameter documentation
- Real-time diagnosis results

## Getting Started

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
DiagnoseRBC/
├── templates/           # HTML templates
│   ├── about.html
│   ├── base.html
│   ├── documentation.html
│   └── index.html
├── data.csv            # Training dataset
├── train_models.py     # Model training script
└── README.md
```

## Blood Parameters Analyzed

The system analyzes various blood parameters including:
- Complete Blood Count (CBC)
- Red Blood Cell indices
- White Blood Cell count
- Platelet count
- Other hematological parameters

## Models

The system implements multiple machine learning models:
- Logistic Regression
- Decision Tree
- Random Forest
- And more...

## Contributing

Created and maintained by Tamim Ahasan Rijon.
Feel free to open issues and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

© 2025 Tamim Ahasan Rijon
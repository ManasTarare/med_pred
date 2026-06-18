# med_pred - Medical Disease Prediction System

A machine learning-based disease prediction system that diagnoses diseases based on patient symptoms and provides comprehensive health recommendations including medications, precautions, diets, and exercise routines.

![med_pred](med_pred_img.png)

## Overview

**med_pred** is an intelligent healthcare application that uses machine learning algorithms to predict potential diseases from a set of reported symptoms. Once a disease is identified, the system provides personalized recommendations for treatment, prevention, and lifestyle management.

## Features

- **Symptom-Based Disease Prediction**: Analyzes patient symptoms to predict potential diseases using trained machine learning models
- **Comprehensive Health Recommendations**: 
  - 💊 Medication suggestions
  - ⚠️ Precautions and safety guidelines
  - 🍎 Dietary recommendations
  - 💪 Personalized workout routines
- **Extensive Medical Database**: Includes descriptions and severity levels for various symptoms
- **User-Friendly Interface**: Simple input-output system for symptom checking

## Project Structure

```
med_pred/
├── med.py                      # Main application script
├── requirements.txt            # Python dependencies
├── Training.csv               # Training dataset for the ML model
├── Symptom-severity.csv       # Symptom severity information
├── description.csv            # Disease descriptions
├── medications.csv            # Medication recommendations
├── precautions_df.csv         # Disease precautions
├── diets.csv                  # Dietary recommendations
├── symtoms_df.csv             # Symptoms database
├── workout_df.csv             # Exercise recommendations
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.6 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ManasTarare/med_pred.git
   cd med_pred
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```bash
python med.py
```

### Input Format

The application prompts users to enter symptoms. Symptoms should be entered one at a time. The system will:

1. Validate the symptom against the symptom database
2. Process all entered symptoms
3. Predict the most likely disease(s)
4. Display detailed information including:
   - Disease name and description
   - Recommended medications
   - Precautions to take
   - Dietary suggestions
   - Recommended workout/exercise routines

## Dataset Description

### Training Data (`Training.csv`)
Contains historical data with symptoms and their associated diseases used to train the prediction model.

### Symptom Data Files
- **symtoms_df.csv**: Complete list of recognized symptoms
- **Symptom-severity.csv**: Severity levels and characteristics of each symptom

### Recommendation Files
- **description.csv**: Detailed descriptions of diseases
- **medications.csv**: Recommended medications for each disease
- **precautions_df.csv**: Precautions and warnings for each disease
- **diets.csv**: Nutritional and dietary recommendations
- **workout_df.csv**: Exercise and fitness recommendations

## How It Works

1. **Input Processing**: User enters symptoms one by one
2. **Symptom Validation**: System validates each symptom against the database
3. **Disease Prediction**: Machine learning model analyzes symptom patterns to predict diseases
4. **Recommendation Retrieval**: System pulls relevant medications, precautions, diet, and workout data
5. **Output Display**: Comprehensive health information is presented to the user

## Machine Learning Model

The system uses supervised learning with the training dataset to build a predictive model. The model is trained to recognize patterns between symptom combinations and disease outcomes, enabling accurate disease prediction based on new symptom inputs.

## Data Files Format

All CSV files follow a structured format with disease-symptom and disease-recommendation mappings:
- Each row represents a mapping between symptoms/recommendations and diseases
- Headers identify the data categories
- Data is normalized for consistent processing

## Requirements

See `requirements.txt` for a complete list of dependencies. Common dependencies include:
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms
- numpy: Numerical computations

## Important Disclaimer

⚠️ **Medical Disclaimer**: This system is designed for educational and informational purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 

**Always consult with a qualified healthcare professional before making any medical decisions.**

## Limitations

- The system predicts diseases based on symptoms alone and may not capture all medical complexities
- Accuracy depends on the completeness and accuracy of symptom input
- Some rare or atypical disease presentations may not be recognized
- Does not account for genetic, environmental, or behavioral factors beyond reported symptoms

## Future Enhancements

- Integration with professional medical databases
- Multi-language support
- Web/mobile interface
- Real-time accuracy metrics and feedback
- Integration with telehealth services
- Support for chronic disease management
- AI model improvements with more diverse training data

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure your contributions follow best practices and include appropriate documentation.

## License

This project is open source and available under an open license. Please see the LICENSE file for details.

## Author

**ManasTarare**

GitHub: [@ManasTarare](https://github.com/ManasTarare)

## Acknowledgments

- Dataset sources for comprehensive symptom and disease information
- Open-source libraries and frameworks that made this project possible
- Healthcare professionals who provided domain expertise

## Support

For issues, questions, or suggestions:
- Open an issue on the [GitHub repository](https://github.com/ManasTarare/med_pred/issues)
- Feel free to fork and improve the project

## Changelog

### Version 1.0
- Initial release
- Core disease prediction functionality
- Symptom validation
- Health recommendations (medications, precautions, diet, workouts)

---

**Last Updated**: June 2026

**Note**: This project is actively maintained. Check back for updates and improvements.

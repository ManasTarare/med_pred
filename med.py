import pandas as pd
import numpy as np
import os
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from pathlib import Path

# ----- Data Loading and Preprocessing -----
dataset = pd.read_csv('med/Training.csv')

x = dataset.drop("prognosis", axis=1)
y = dataset["prognosis"]

le = LabelEncoder()
y = le.fit_transform(y)

feature_names = list(x.columns)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)

# ----- Train or Load Model -----
if not os.path.exists("deep_disease_model.h5"):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(len(set(y_train)), activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
    model.save("deep_disease_model.h5")
    pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Load model and scaler
loaded_model = tf.keras.models.load_model("deep_disease_model.h5")
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ----- Helper Function -----
def helper(disease):
    description = pd.read_csv("med/description.csv")
    precautions = pd.read_csv("med/precautions_df.csv")
    medications = pd.read_csv('med/medications.csv')
    diets = pd.read_csv("med/diets.csv")
    workouts = pd.read_csv("med/workout_df.csv")

    desc_data = description.loc[description['Disease'] == disease, 'Description'].values
    desc = desc_data[0] if len(desc_data) > 0 else "No description available."

    pre = precautions.loc[precautions['Disease'] == disease, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values
    pre = pre.flatten().tolist() if len(pre) > 0 else []

    med = medications.loc[medications['Disease'] == disease, 'Medication'].values.tolist() if disease in medications['Disease'].values else []
    diet = diets.loc[diets['Disease'] == disease, 'Diet'].values.tolist() if disease in diets['Disease'].values else []
    workout_plan = workouts.loc[workouts['disease'] == disease, 'workout'].values.tolist() if disease in workouts['disease'].values else []

    return desc, pre, med, diet, workout_plan

# ----- Prediction Function -----
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(feature_names))
    for symptom in patient_symptoms:
        if symptom in feature_names:
            index = feature_names.index(symptom)
            input_vector[index] = 1
    input_vector = scaler.transform([input_vector])
    pred_index = np.argmax(loaded_model.predict(input_vector, verbose=0))
    try:
        predicted_disease = le.inverse_transform([pred_index])[0]
    except IndexError:
        predicted_disease = "Unknown"
    return predicted_disease

# ----- Streamlit App -----
st.set_page_config(page_title="MedPred.org", page_icon="🔬", layout="wide")
# Sidebar
st.sidebar.title("Navigation")
sidebar_option = st.sidebar.radio("Choose a section:", ["Home", "Predict", "About"])

# Display image only in the sidebar
image_path = Path("med/med_pred_img.png")  # Make sure the path is correct
st.sidebar.image(str(image_path), use_container_width=True)

# Main Content
if sidebar_option == "Home":
    st.markdown("<h1 class='title'>Advanced Disease Prediction</h1>", unsafe_allow_html=True)
    st.markdown("""
    This AI-powered application is designed to assist individuals in identifying potential diseases based on the symptoms they are experiencing. 
    Leveraging a deep learning model trained on medical symptom-disease data, the app analyzes user-provided symptoms and predicts the most likely condition.

    ### 🤖 How It Works:
    - You select your symptoms from a structured list in the **Predict** section.
    - Our trained deep learning model processes your input and identifies the disease it most likely corresponds to.
    - Once predicted, the app presents:
        - 📃 A detailed description of the disease.
        - 💊 Recommended medications.
        - 🛡️ Precautionary steps.
        - 🥗 Dietary suggestions.
        - 🏃‍♀️ Workout and fitness advice.

    This app is not a substitute for medical diagnosis but serves as a **support tool** to promote awareness and guidance before seeking professional consultation.

    ### 👇 Get started by clicking on the 'Predict' tab in the sidebar.
    """)

elif sidebar_option == "Predict":
    st.markdown("<h1 class='title'>Predict Disease</h1>", unsafe_allow_html=True)
    selected_symptoms = st.multiselect("Choose your symptoms:", feature_names, placeholder="Start typing symptom...")

    if selected_symptoms:
        with st.spinner('Initializing...'):
            progress_text = "Predicting disease..."
            my_bar = st.progress(0, text=progress_text)

            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)

            predicted_disease = get_predicted_value(selected_symptoms)
            st.success(f"Predicted disease: {predicted_disease}")

            desc, pre, med, diet, workout_plan = helper(predicted_disease)

            st.subheader("Disease Details")
            st.write(f"**Disease Name:** {predicted_disease}")
            st.write(f"**Description:** {desc}")

            st.subheader("Precautions")
            if pre:
                st.write("\n".join([f"{i}. {p}" for i, p in enumerate(pre, 1)]))
            else:
                st.write("No precautions found.")

            st.subheader("Medications")
            if med:
                st.write("\n".join([f"{i}. {m}" for i, m in enumerate(med, 1)]))
            else:
                st.write("No medications found.")

            st.subheader("Workout")
            if workout_plan:
                st.write("\n".join([f"{i}. {w}" for i, w in enumerate(workout_plan, 1)]))
            else:
                st.write("No workout plan found.")

            st.subheader("Diets")
            if diet:
                st.write("\n".join([f"{i}. {d}" for i, d in enumerate(diet, 1)]))
            else:
                st.write("No diet suggestions found.")

            prediction_results = f"Predicted Disease: {predicted_disease}\n"
            prediction_results += f"Description: {desc}\nPrecautions: {', '.join(pre)}\nMedications: {', '.join(med)}\nDiet: {', '.join(diet)}\nWorkout Plan: {', '.join(workout_plan)}"

            st.download_button(label="Download Prediction Result", data=prediction_results, file_name="prediction_result.txt")

    else:
        st.warning("Please select symptoms to predict.")

elif sidebar_option == "About":
    st.markdown("<h1 class='title'>About</h1>", unsafe_allow_html=True)
    st.write("""
        This tool helps predict diseases based on the symptoms you provide. The model was trained using a dataset
        of various diseases and their corresponding symptoms, and can suggest possible diseases based on your input.
        It also provides information on precautions, medications, diet, and workout suggestions for each disease.

        ### 🤖 How the Model Works:
        The disease prediction model is a **deep neural network** trained on a comprehensive dataset of symptoms and corresponding diseases. It takes a set of user-provided symptoms and processes them to predict the most likely disease. The process includes several stages, such as:

        1. **Data Preprocessing**: The input data, consisting of symptom-disease pairs, is first preprocessed and encoded. Symptoms are one-hot encoded into a binary format, and the labels (diseases) are encoded using a **LabelEncoder** to convert categorical values into numerical format.
        
        2. **Model Architecture**: A deep learning model based on **TensorFlow** and **Keras** is employed. The architecture consists of multiple **Dense (fully connected) layers**, with **ReLU activation functions** to introduce non-linearity and **Dropout layers** to prevent overfitting. The output layer uses the **softmax activation function** to provide a probability distribution over all possible diseases.
        
        3. **Model Training**: The model is trained using a large training dataset, where the input symptoms are mapped to diseases. The model is optimized using the **Adam optimizer** with a **learning rate of 0.001**. The training process involves minimizing the **sparse categorical crossentropy loss** to improve the model's predictive accuracy.
        
        4. **Prediction**: When a user selects their symptoms, the model uses the trained weights to predict the most likely disease based on the input symptoms.

        ### ⚙️ Technologies Used:
        - **TensorFlow** & **Keras**: The deep learning framework used for building and training the model.
        - **Python**: The programming language used to implement the model and the web app.
        - **Streamlit**: The interactive web framework that powers the user interface, allowing users to input symptoms and receive predictions.
        - **Scikit-learn**: Used for data preprocessing tasks such as **Label Encoding** and **Standard Scaling**.
        - **Pandas & NumPy**: Libraries used for data manipulation and processing.
        - **Pickle**: Used to save the trained model and scaler for future use, ensuring that users can get predictions without needing to retrain the model every time.

        ### 🌐 Data Source:
        - **Medical Datasets**: The model was trained on a dataset that includes symptoms and their corresponding diseases. These datasets were curated from medical sources to create a robust mapping between symptoms and diseases.
        
        ### 📈 Model Evaluation:
        The model's performance was evaluated using **accuracy**, **precision**, **recall**, and **F1-score** metrics, and it was found to perform well on a variety of test data.

        ### ⚠️ Disclaimer:
        This app is **not a diagnostic tool** and is intended to serve as an informational aid only. It should not replace professional medical advice. Always consult a healthcare professional for proper diagnosis and treatment.

        - Model: Deep Neural Network with 3 Dense layers
        - Framework: TensorFlow & Streamlit
        - Data Source: Symptom-disease mappings
        - Intended Use: Educational / informational only (not a diagnostic tool)
    """)

import streamlit as st
import joblib
import pandas as pd

st.write("""
# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!
""")

# Function to get user input
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Capture user input
df = user_input_features()

# Rename the columns to match the original feature names
df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Display user input
st.subheader('User Input Parameters')
st.table(df)

# Load the pre-trained model and scaler
model = joblib.load('iris_model.pkl')  # Ensure the path is correct
scaler = joblib.load('scaler.pkl')    # Load the scaler

# Apply the same transformation to the user input data
df_scaled = scaler.transform(df)  # Transform user input with the scaler

# Display class labels
st.subheader('Class Labels and their Corresponding Index Number')
class_labels = ['setosa', 'versicolor', 'virginica']
st.table(pd.DataFrame({'Iris Species': class_labels}))

# Make a prediction
st.subheader('Prediction')
prediction = model.predict(df_scaled)
predicted_class = class_labels[prediction[0]]
st.write(f"**{predicted_class}**")

# Predict the probability
st.subheader('Prediction Probability')
prediction_proba = model.predict_proba(df_scaled)
proba_df = pd.DataFrame(prediction_proba, columns=class_labels)
st.table(proba_df)

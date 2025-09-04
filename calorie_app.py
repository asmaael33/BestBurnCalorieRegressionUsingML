# 📦 Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# 📁 Step 2: Load or simulate dataset
# You can replace this with a real dataset later
np.random.seed(42)
n_samples = 500
data = pd.DataFrame({
    'Age': np.random.randint(18, 60, n_samples),
    'Weight': np.random.randint(50, 100, n_samples),
    'Height': np.random.randint(150, 200, n_samples),
    'Duration': np.random.randint(10, 60, n_samples),  # minutes
    'Gender': np.random.choice(['Male', 'Female'], n_samples)
})

# Simulate calories burned (simplified formula)
data['Calories'] = (
    0.05 * data['Weight'] +
    0.03 * data['Height'] +
    0.1 * data['Duration'] +
    (data['Gender'] == 'Male') * 5 +
    np.random.normal(0, 10, n_samples)
)

# 🔄 Step 3: Preprocess
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
X = data[['Age', 'Weight', 'Height', 'Duration', 'Gender']]
y = data['Calories']

# 🧠 Step 4: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

# 📊 Step 5: Evaluate
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")

# 🔮 Step 6: Predict function
def predict_calories(age, weight, height, duration, gender):
    gender_val = 1 if gender.lower() == 'male' else 0
    input_data = pd.DataFrame([[age, weight, height, duration, gender_val]],
                              columns=['Age', 'Weight', 'Height', 'Duration', 'Gender'])
    prediction = model.predict(input_data)[0]
    print(f"🔥 Estimated Calories Burned: {prediction:.2f} kcal")

# 🧪 Example usage
predict_calories(age=43, weight=67, height=169, duration=45, gender='Female')


# 📦 Add these imports
from ipywidgets import interact, IntSlider, Dropdown

# 🔮 Wrap prediction in a widget
def interactive_calorie_predictor(age, weight, height, duration, gender):
    gender_val = 1 if gender == 'Male' else 0
    input_df = pd.DataFrame([[age, weight, height, duration, gender_val]],
                            columns=['Age', 'Weight', 'Height', 'Duration', 'Gender'])
    prediction = model.predict(input_df)[0]
    print(f"🔥 Estimated Calories Burned: {prediction:.2f} kcal")

# 🎛️ Create sliders and dropdowns
interact(interactive_calorie_predictor,
         age=IntSlider(min=18, max=65, step=1, value=30),
         weight=IntSlider(min=40, max=120, step=1, value=70),
         height=IntSlider(min=140, max=200, step=1, value=170),
         duration=IntSlider(min=10, max=90, step=5, value=30),
         gender=Dropdown(options=['Male', 'Female'], value='Female'))


import folium

# 🧪 Simulate locations and predictions
locations = [
    {'lat': 34.020882, 'lon': -6.841650, 'age': 28, 'weight': 65, 'height': 170, 'duration': 40, 'gender': 'Female'},
    {'lat': 33.9716, 'lon': -6.8498, 'age': 35, 'weight': 80, 'height': 180, 'duration': 60, 'gender': 'Male'},
]

# 🗺️ Create map
m = folium.Map(location=[33.97, -6.85], zoom_start=12)

# 🔄 Add markers
for loc in locations:
    gender_val = 1 if loc['gender'] == 'Male' else 0
    input_df = pd.DataFrame([[loc['age'], loc['weight'], loc['height'], loc['duration'], gender_val]],
                            columns=['Age', 'Weight', 'Height', 'Duration', 'Gender'])
    prediction = model.predict(input_df)[0]
    popup_html = f"""
    <b>Calories Burned:</b> {prediction:.2f} kcal<br>
    <b>Age:</b> {loc['age']}<br>
    <b>Weight:</b> {loc['weight']} kg<br>
    <b>Duration:</b> {loc['duration']} min
    """
    folium.Marker(
        location=[loc['lat'], loc['lon']],
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(m)

m



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# 🔁 Load trained model (or train it here)
# For demo, we'll train a simple model
def train_model():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'Age': np.random.randint(18, 60, n),
        'Weight': np.random.randint(50, 100, n),
        'Height': np.random.randint(150, 200, n),
        'Duration': np.random.randint(10, 60, n),
        'Gender': np.random.choice([0, 1], n)  # 0 = Female, 1 = Male
    })
    df['Calories'] = (
        0.05 * df['Weight'] +
        0.03 * df['Height'] +
        0.1 * df['Duration'] +
        df['Gender'] * 5 +
        np.random.normal(0, 10, n)
    )
    model = LinearRegression()
    model.fit(df[['Age', 'Weight', 'Height', 'Duration', 'Gender']], df['Calories'])
    return model

model = train_model()

# 🌐 Streamlit UI
st.title("🔥 Calorie Burn Predictor")

age = st.slider("Age", 18, 65, 30)
weight = st.slider("Weight (kg)", 40, 120, 70)
height = st.slider("Height (cm)", 140, 200, 170)
duration = st.slider("Exercise Duration (minutes)", 10, 90, 30)
gender = st.radio("Gender", ['Female', 'Male'])
gender_val = 1 if gender == 'Male' else 0

# 🔮 Prediction
input_df = pd.DataFrame([[age, weight, height, duration, gender_val]],
                        columns=['Age', 'Weight', 'Height', 'Duration', 'Gender'])
prediction = model.predict(input_df)[0]

st.subheader(f"Estimated Calories Burned: {prediction:.2f} kcal")


!streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py
import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from user_db import USER_DB


# -------------------------------
# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.nickname = ""
    st.session_state.history = pd.DataFrame(columns=["Date", "Calories"])

# -------------------------------
# Login screen
def login():
    st.title("🏋️‍♀️ Gym Calorie Tracker Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    nickname = st.text_input("Nickname")

    if st.button("Login"):
        user = USER_DB.get(username)
        if user and user["password"] == password and user["nickname"] == nickname:
            st.session_state.logged_in = True
            st.session_state.nickname = nickname
            st.success(f"Welcome back, {nickname}!")
        else:
            st.error("Invalid credentials or nickname.")

# -------------------------------
# Main app
def main_app():
    st.title(f"🔥 Welcome {st.session_state.nickname}!")
    st.markdown("Enter your gym session details to calculate calories burned:")

    # Input sliders
    age = st.slider("Age", 13, 90, 30)
    weight = st.slider("Weight (kg)", 30, 150, 70)
    duration = st.slider("Duration (hours)", 0.1, 3.0, 1.0, step=0.1)
    met = st.slider("MET Value", 1.0, 15.0, 6.0, step=0.5)
    gender = st.selectbox("Gender", ["Female", "Male"])
    session_date = st.date_input("Session Date", value=datetime.date.today())

    # Gender factor
    gender_factor = 0.9 if gender == "Female" else 1.0

    # Calculate calories
    calories = round(met * weight * duration * gender_factor, 2)
    st.subheader(f"✅ Calories Burned: {calories} kcal")

    # Save to history
    if st.button("Save Session"):
        new_entry = pd.DataFrame({"Date": [session_date], "Calories": [calories]})
        st.session_state.history = pd.concat([st.session_state.history, new_entry], ignore_index=True)
        st.success("Session saved!")

    # Historical graph
    st.markdown("### 📅 View History")
    selected_date = st.date_input("Filter by date", value=datetime.date.today())
    filtered = st.session_state.history[st.session_state.history["Date"] == selected_date]

    if not filtered.empty:
        st.line_chart(filtered.set_index("Date")["Calories"])
    else:
        st.info("No data for selected date.")

# -------------------------------
# Run app
if not st.session_state.logged_in:
    login()
else:
    main_app()

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import datetime

# Load credentials from config.yaml
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Login widget
authenticator.login()

# Access control
if st.session_state["authentication_status"]:
    authenticator.logout("Logout", "sidebar")
    st.title(f"🔥 Welcome {st.session_state['name']}!")
    st.markdown("Enter your gym session details to calculate calories burned:")

    # Initialize history
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=["Date", "Calories"])

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

elif st.session_state["authentication_status"] is False:
    st.error("Username/password is incorrect")
elif st.session_state["authentication_status"] is None:
    st.warning("Please enter your username and password")

import streamlit as st

# Title and description
st.title("💓 Fat-Burning Heart Rate Zone Calculator")
st.markdown("""
This app helps you find your optimal fat-burning heart rate zone based on your age.  
The zone is **60% to 80%** of your maximum heart rate, which is calculated as `220 - age`.
""")

# Age input
age = st.slider("Select your age", min_value=13, max_value=90, value=30)

# Calculate heart rate zone
max_hr = 220 - age
lower_bound = int(max_hr * 0.60)
upper_bound = int(max_hr * 0.80)

# Display results
st.subheader(f"Results for Age {age}")
st.write(f"- **Max Heart Rate**: {max_hr} bpm")
st.write(f"- **Fat-Burning Zone**: {lower_bound} – {upper_bound} bpm")
st.success("🏃 Aim to stay within this range during cardio for optimal fat metabolism.")

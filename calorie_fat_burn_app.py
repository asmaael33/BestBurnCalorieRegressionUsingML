import streamlit as st

# Title and description
st.title("🔥 Calorie Burn & Fat Loss Estimator")
st.markdown("""
Estimate how many calories you burn during physical activity, and convert that into grams of fat lost.  
The calculation uses your **age**, **weight**, **duration**, **gender**, and **MET value** (activity intensity).
""")

# Input sliders and dropdowns
age = st.slider("Age", min_value=13, max_value=90, value=30)
weight = st.slider("Weight (kg)", min_value=30, max_value=150, value=70)
duration = st.slider("Duration (hours)", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
met = st.slider("MET Value", min_value=1.0, max_value=15.0, value=6.0, step=0.5)
gender = st.selectbox("Gender", options=["Female", "Male"])

# Gender factor
gender_factor = 0.9 if gender == "Female" else 1.0

# Calculate calories burned
calories_burned = round(met * weight * duration * gender_factor, 2)

# Display results
st.subheader("🔢 Results")
st.write(f"- **Calories Burned**: `{calories_burned}` kcal")

# Button to convert to fat grams
if st.button("Convert to Fat Burned (g)"):
    fat_grams = round(calories_burned / 9, 2)
    st.success(f"🧪 Estimated Fat Burned: **{fat_grams} grams**")

# Optional tip
st.info("💡 1 gram of fat ≈ 9 kcal. Use MET values from known activities like walking (3.5), running (9.8), swimming (6.0), etc.")

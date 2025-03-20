import joblib
import streamlit as st


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

# Streamlit UI
st.title("🔍 Text Classification App")
st.markdown("Enter text below and classify it using the best trained model.")

# Text Input
text_input = st.text_area("Enter your text here:")
if st.button("Classify Text"):
    if text_input:
        prediction = model.predict([text_input])[0]
        st.success(f"Predicted Class: {prediction}")
    else:
        st.warning("Please enter some text to classify.")

st.markdown("---")
st.caption("✨ Built with Streamlit and Scikit-learn.")

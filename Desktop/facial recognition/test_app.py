import streamlit as st

st.title("🏋️ Arnold Facial Recognition Test")
st.write("This is a test to verify Streamlit Cloud deployment is working!")

if st.button("Test Button"):
    st.success("✅ Deployment is working!")
    st.write("Now we can deploy the full app!")

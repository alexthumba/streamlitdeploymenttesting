# Import streamlit
import streamlit as st

# Write some text
st.write("Hello, this is a simple streamlit app")

# Create a button
button = st.button("Click me")

# Display a message when the button is clicked
if button:
    st.write("You clicked the button!")
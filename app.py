# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="My Simple Streamlit App",
    page_icon="🚀",
    layout="wide"
)

# Title and description
st.title("🚀 My Simple Streamlit App")
st.markdown("Welcome to this interactive demo!")

# Sidebar
st.sidebar.header("Settings")
user_name = st.sidebar.text_input("Enter your name:", "John Doe")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Basic Controls")
    
    # Slider
    age = st.slider("Select your age:", 0, 100, 25)
    
    # Selectbox
    favorite_color = st.selectbox(
        "Choose your favorite color:",
        ["Red", "Blue", "Green", "Yellow", "Purple"]
    )
    
    # Button
    if st.button("Say Hello!"):
        st.success(f"Hello {user_name}! 👋")

with col2:
    st.header("Data Visualization")
    
    # Generate sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x': range(1, 101),
        'y': np.random.randn(100).cumsum()
    })
    
    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot(data['x'], data['y'])
    ax.set_title("Random Walk Data")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# Display user information
st.header("Your Information")
st.write(f"**Name:** {user_name}")
st.write(f"**Age:** {age}")
st.write(f"**Favorite Color:** {favorite_color}")

# File uploader
st.header("File Upload")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    st.subheader("Data Summary")
    st.write(f"Shape: {df.shape}")
    st.write("Column types:")
    st.write(df.dtypes)

# Expander for additional info
with st.expander("ℹ️ About this app"):
    st.write("""
    This is a simple Streamlit app demonstrating:
    - Basic input widgets
    - Data visualization
    - File uploading
    - Layout management
    - Interactive components
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")

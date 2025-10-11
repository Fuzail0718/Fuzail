import streamlit as st

st.title("🧮 Simple Calculator")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    num1 = st.number_input("Enter first number:", value=0.0)

with col2:
    operation = st.selectbox("Select operation:", 
                           ["+", "-", "×", "÷"])

with col3:
    num2 = st.number_input("Enter second number:", value=0.0)

# Calculate button
if st.button("Calculate"):
    try:
        if operation == "+":
            result = num1 + num2
        elif operation == "-":
            result = num1 - num2
        elif operation == "×":
            result = num1 * num2
        elif operation == "÷":
            if num2 == 0:
                st.error("Error: Cannot divide by zero!")
                result = None
            else:
                result = num1 / num2
        
        if result is not None:
            st.success(f"**Result:** {num1} {operation} {num2} = **{result}**")
            
    except Exception as e:
        st.error(f"Error: {e}")

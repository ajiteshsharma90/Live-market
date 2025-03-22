import streamlit as st
import plotly.graph_objects as go

st.write("Testing minimal image generation...")

# Create a simple figure.
fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6])])
fig.update_layout(title="Test Chart")

try:
    st.write("Attempting to generate image using to_image()...")
    # Generate image bytes in-memory
    image_bytes = fig.to_image(format="png", engine="kaleido")
    
    # Display the image
    st.image(image_bytes, caption="Generated Test Chart", use_column_width=True)
    st.write("Image generation successful!")
except Exception as e:
    st.error(f"Error generating image: {e}")

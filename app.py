import os
import gradio as gr
import subprocess
import threading
import time

# Set up the Streamlit process
def run_streamlit():
    subprocess.run(["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"])

# Start Streamlit in a separate thread
threading.Thread(target=run_streamlit, daemon=True).start()

# Allow Streamlit some time to start
time.sleep(5)

# Create a Gradio interface that embeds Streamlit
demo = gr.Interface(
    fn=lambda x: x,
    inputs=None,
    outputs=None,
    title="Document QA with RAG",
    description="This application is running on Streamlit. Please click the link below.",
    article="Access the Streamlit app directly at: [Streamlit App](http://localhost:8501)"
)

# Launch the Gradio interface
demo.launch()
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from tensorflow import keras
from keras.models import load_model

def load_all_resources():
    st.toast("Attempting to load model and data...")
    try:
        st.toast("1/4: Loading the Keras model: MLP_model.keras")
        model = load_model('MLP_model.keras')
        st.toast("Keras model loaded successfully.")

        st.toast("2/4: Loading selector.pkl...")
        with open('selector.pkl', 'rb') as f:
            selector = pkl.load(f)
        st.toast("selector.pkl loaded successfully.")

        st.toast("3/4: Loading scaler.pkl...")
        with open('scaler.pkl', 'rb') as f:
            scaler = pkl.load(f)
        st.toast("scaler.pkl loaded successfully.")

        st.toast("4/4: Loading medians.pkl...")
        with open('medians.pkl', 'rb') as f:
            medians = pkl.load(f)
        st.toast("medians.pkl loaded successfully.")

        return model, selector, scaler, medians
    except FileNotFoundError as e:
        st.error(f"A file was not found: {e}. Please check the filename and location.")
        st.warning("Ensure all model and preprocessing files are in the same directory as app.py.")
        return None, None, None, None
    except Exception as e:
        st.error("An unexpected error occurred during file loading.")
        st.exception(e)
        return None, None, None, None

st.set_page_config(page_title="Wind Turbine Failure Predictor", layout="wide")
st.title("Wind Turbine Predictive Maintenance App")
st.write("Upload a CSV/XLSX file with new sensor data to get a failure prediction.")
st.markdown("---")

model, selector, scaler, medians = load_all_resources()

if model is not None and selector is not None and scaler is not None and medians is not None:
    st.success("All resources loaded successfully! The app is ready for your input.")

    st.markdown("### Step 1: Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or XLSX file", type=["csv", 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                st.stop()
            
            st.write("### Raw Data")
            st.dataframe(df)

            new_df = df.fillna(medians)
            new_df_selected = selector.transform(new_df)
            new_df_scaled = pd.DataFrame(scaler.transform(new_df_selected))
            
            y_pred_proba = model.predict(new_df_scaled).flatten()
            y_pred_hard = (y_pred_proba > 0.5).astype(int)
            
            df['Prediction'] = y_pred_hard
            
            st.write("### Predictions")
            st.dataframe(df)
            st.markdown("---")
            
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv_output,
                file_name='predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"An error occurred during processing: {e}. Please check your CSV file format.")
else:
    # This block is executed if resource loading failed
    st.warning("Model files are not loaded. Please train your model and save the required files first.")


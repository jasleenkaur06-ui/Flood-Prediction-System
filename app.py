
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load("flood_model.pkl")
features = joblib.load("features.pkl")

# ==========================================
# 3. FEATURE RANGES
# ==========================================

feature_ranges = {
    "MonsoonIntensity": 16,
    "TopographyDrainage": 18,
    "RiverManagement": 16,
    "Deforestation": 17,
    "Urbanization": 17,
    "ClimateChange": 17,
    "DamsQuality": 16,
    "Siltation": 16,
    "AgriculturalPractices": 16,
    "Encroachments": 18,
    "IneffectiveDisasterPreparedness": 16,
    "DrainageSystems": 17,
    "CoastalVulnerability": 17,
    "Landslides": 16,
    "Watersheds": 16,
    "DeterioratingInfrastructure": 17,
    "PopulationScore": 18,
    "WetlandLoss": 19,
    "InadequatePlanning": 16,
    "PoliticalFactors": 16
}

# ==========================================
# 4. DISPLAY NAMES
# ==========================================

display_names = {
    "MonsoonIntensity": "Monsoon Intensity",
    "TopographyDrainage": "Topography Drainage",
    "RiverManagement": "River Management",
    "Deforestation": "Deforestation",
    "Urbanization": "Urbanization",
    "ClimateChange": "Climate Change",
    "DamsQuality": "Dams Quality",
    "Siltation": "Siltation",
    "AgriculturalPractices": "Agricultural Practices",
    "Encroachments": "Encroachments",
    "IneffectiveDisasterPreparedness": "Ineffective Disaster Preparedness",
    "DrainageSystems": "Drainage Systems",
    "CoastalVulnerability": "Coastal Vulnerability",
    "Landslides": "Landslides",
    "Watersheds": "Watersheds",
    "DeterioratingInfrastructure": "Deteriorating Infrastructure",
    "PopulationScore": "Population Score",
    "WetlandLoss": "Wetland Loss",
    "InadequatePlanning": "Inadequate Planning",
    "PoliticalFactors": "Political Factors"
}

# ==========================================
# 5. FEATURE CATEGORIES
# ==========================================

categories = {
    "🌦️ Environmental Factors": [
        "MonsoonIntensity",
        "ClimateChange",
        "Deforestation",
        "Landslides",
        "WetlandLoss",
        "CoastalVulnerability"
    ],

    "🌊 Water & Infrastructure": [
        "TopographyDrainage",
        "RiverManagement",
        "DamsQuality",
        "Siltation",
        "DrainageSystems",
        "Watersheds",
        "DeterioratingInfrastructure"
    ],

    "🏙️ Human Activities": [
        "Urbanization",
        "AgriculturalPractices",
        "Encroachments",
        "PopulationScore"
    ],

    "📋 Planning & Governance": [
        "IneffectiveDisasterPreparedness",
        "InadequatePlanning",
        "PoliticalFactors"
    ]
}

# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("🌊 Flood Prediction")
st.sidebar.caption("Machine Learning Based System")

st.sidebar.divider()

st.sidebar.subheader("About Project")

st.sidebar.write(
    "A machine learning system that estimates "
    "flood probability scores using environmental, "
    "geographical and human-related factors."
)

st.sidebar.info("Model: Linear Regression")

st.sidebar.divider()

st.sidebar.caption("Dataset: Flood Prediction Dataset")
st.sidebar.caption("Task: Supervised Learning - Regression")

# ==========================================
# 7. MAIN DASHBOARD
# ==========================================

st.title("🌊 Flood Prediction Dashboard")

st.write(
    "Enter environmental and geographical details "
    "to generate an estimated flood probability score."
)

st.divider()

# ==========================================
# 8. INPUT FORM
# ==========================================

user_inputs = {}

with st.form("flood_prediction_form"):

    for category, category_features in categories.items():

        st.subheader(category)

        col1, col2 = st.columns(2)

        for i, feature in enumerate(category_features):

            if feature not in features:
                continue

            current_col = col1 if i % 2 == 0 else col2

            with current_col:

                user_inputs[feature] = st.number_input(
                    display_names[feature],
                    min_value=0,
                    max_value=feature_ranges[feature],
                    value=5,
                    step=1,
                    help=f"Allowed range: 0 to {feature_ranges[feature]}. This is a dataset-based score, NOT actual rainfall in mm or number of rainy days.",
key=feature
                )
                

        st.divider()

    predict_button = st.form_submit_button(
        "🔍 Predict Flood Probability",
        type="primary",
        use_container_width=True
    )

# ==========================================
# 9. PREDICTION & VISUALIZATION
# ==========================================

if predict_button:

    input_data = pd.DataFrame(
        [[user_inputs[f] for f in features]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    st.header("📊 Flood Probability Analysis")

    st.success("Prediction generated successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Flood Probability Score",
            f"{prediction * 100:.2f}%"
        )

    with col2:
        st.write("Prediction Score")
        st.progress(float(max(0, min(1, prediction))))

    st.info(
        "This is a model-estimated score based on the dataset. "
        "It is not an official flood warning or confirmed "
        "real-world risk assessment."
    )

    # Input visualization
    st.divider()

    st.subheader("📊 Input Factors Visualization")

    input_chart = pd.DataFrame({
        "Factor": [display_names[f] for f in features],
        "Input Value": [user_inputs[f] for f in features]
    })

    st.bar_chart(
        data=input_chart,
        x="Factor",
        y="Input Value",
        horizontal=True
    )

    st.caption(
        "This chart compares all 20 values entered by the user. "
        "It does not represent individual feature importance."
    )

    
    # Download Prediction Report

    
    # Save all predictions in one CSV file

    import os
    from datetime import datetime

    history_file = "prediction_history.csv"

    report_data = {
        "Date & Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    for f in features:
        report_data[display_names[f]] = user_inputs[f]

    report_data["Predicted Flood Probability"] = prediction

    new_record = pd.DataFrame([report_data])

    if os.path.exists(history_file):
        new_record.to_csv(
            history_file,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_record.to_csv(history_file, index=False)

    st.success("Prediction saved successfully!")

    with open(history_file, "rb") as file:
        st.download_button(
            "📥 Download Complete Prediction History",
            data=file,
            file_name="prediction_history.csv",
            mime="text/csv"
        )



# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("📈 Model Performance Analysis")

st.write(
    "Performance of the Linear Regression model "
    "on the held-out test dataset."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "0.01579")

with col2:
    st.metric("RMSE", "0.02008")

with col3:
    st.metric("R² Score", "0.84488")

with st.expander("Understanding the Evaluation Metrics"):

    st.write(
        "**MAE:** Average absolute difference "
        "between actual and predicted values."
    )

    st.write(
        "**RMSE:** Measures prediction errors "
        "with greater penalty for larger errors."
    )

    st.write(
        "**R² Score:** Measures the proportion of "
        "variation explained by the model on test data."
    )

st.caption(
    "Evaluation: 80:20 train-test split, random_state=42."
)



# ==========================================
# 11. FEATURE COEFFICIENT ANALYSIS
# ==========================================

st.divider()

st.header("🎯 Feature Influence Analysis")

st.write(
    "This analysis examines how each input feature "
    "is associated with the predicted flood probability score "
    "according to the Linear Regression model."
)

# Create coefficient dataframe
coefficient_data = pd.DataFrame({
    "Factor": [display_names[f] for f in features],
    "Coefficient": model.coef_,
    "Feature Range": [feature_ranges[f] for f in features]
})

# Estimate effect across each feature's full dataset range
coefficient_data["Range Effect"] = (
    coefficient_data["Coefficient"] *
    coefficient_data["Feature Range"]
)

# Sort by absolute range effect
coefficient_data["Absolute Effect"] = (
    coefficient_data["Range Effect"].abs()
)

coefficient_data = coefficient_data.sort_values(
    by="Absolute Effect",
    ascending=True
)

# Display chart
st.subheader("Range-Adjusted Coefficient Comparison")

st.bar_chart(
    coefficient_data,
    x="Range Effect",
    y="Factor",
    horizontal=True
)

st.caption(
    "Range Effect = Linear Regression coefficient × "
    "observed feature range. Positive values indicate a "
    "positive relationship, while negative values indicate "
    "a negative relationship."
)

# Display coefficient table
with st.expander("View Detailed Feature Coefficients"):

    st.dataframe(
        coefficient_data[
            ["Factor", "Coefficient", "Feature Range", "Range Effect"]
        ].sort_values(
            by="Range Effect",
            key=abs,
            ascending=False
        ),
        use_container_width=True
    )

st.info(
    "These values describe the fitted model's linear relationships. "
    "They should not be interpreted as causal effects or "
    "independent real-world flood risk measurements."
)
# Prediction History

# Prediction History

import os

st.divider()
st.subheader("📋 Prediction History")

history_file = "prediction_history.csv"

if os.path.exists(history_file):
    history_df = pd.read_csv(history_file)

    st.write("Total Predictions:", len(history_df))

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if not history_df.empty:

        st.subheader("🗑️ Delete Specific Prediction")

        selected_index = st.selectbox(
            "Choose the prediction you want to delete:",
            options=history_df.index.tolist(),
            format_func=lambda i: (
                f"Record {i + 1} | "
                f"{history_df.loc[i, 'Date & Time']} | "
                f"{float(history_df.loc[i, 'Predicted Flood Probability']) * 100:.2f}%"
            )
        )

        confirm_delete = st.checkbox(
            "I confirm that I want to delete this prediction."
        )

        if st.button(
            "Delete Selected Prediction",
            disabled=not confirm_delete
        ):
            history_df = history_df.drop(index=selected_index)

            history_df.to_csv(history_file, index=False)

            st.success("Selected prediction deleted!")
            st.rerun()

else:
    st.info("No prediction history available yet.")
    
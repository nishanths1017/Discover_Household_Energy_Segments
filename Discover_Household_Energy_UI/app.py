import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Household Energy Segments",
    page_icon="⚡",
    layout="wide"
)

DATA_PATH = "data/household_energy_segments_final.csv"
SCALER_PATH = "models/scaler.pkl"
PCA_PATH = "models/pca.pkl"
KMEANS_PATH = "models/kmeans.pkl"

df = pd.read_csv(DATA_PATH)

scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)
kmeans = joblib.load(KMEANS_PATH)

df["DateTime"] = pd.to_datetime(df["DateTime"])

segment_labels = {
    0: "Moderate Stable Consumption",
    1: "High Consumption High Demand",
    2: "Low Consumption High Variability",
    3: "Very Low Consumption"
}

segment_descriptions = {
    0: "Moderate energy use with relatively stable power behavior.",
    1: "Higher daily energy use and higher maximum power demand.",
    2: "Lower energy use with more variable and peak-oriented consumption.",
    3: "Very low daily energy consumption compared with the other segments."
}

recommendations = {
    0: "General energy-efficiency recommendations and routine monitoring.",
    1: "Peak-demand management, load shifting, and targeted efficiency programs.",
    2: "Peak-event monitoring, load smoothing, and time-of-use recommendations.",
    3: "Low-usage monitoring and lightweight conservation recommendations."
}

cluster_features = [
    "Daily_Energy_kWh",
    "Max_Active_Power",
    "Peak_to_Average_Ratio",
    "Power_Variability",
    "Submeter_1_Ratio",
    "Submeter_2_Ratio",
    "Submeter_3_Ratio",
    "Reactive_to_Active_Ratio",
    "Avg_Voltage"
]

st.title("⚡ Discover Household Energy Segments")
st.write("Daily household electricity consumption pattern analysis")

st.divider()

st.subheader("📊 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Complete Days", f"{len(df):,}")

with col2:
    st.metric("Segments", df["Segment"].nunique())

with col3:
    st.metric(
        "Average Energy",
        f"{df['Daily_Energy_kWh'].mean():.2f} kWh"
    )

with col4:
    st.metric(
        "Maximum Energy",
        f"{df['Daily_Energy_kWh'].max():.2f} kWh"
    )

st.divider()

st.subheader("📌 Discovered Consumption Segments")

segment_counts = df["Segment"].value_counts().sort_index()

cards = st.columns(4)

for segment, card in enumerate(cards):
    count = int(segment_counts.get(segment, 0))
    percentage = count / len(df) * 100

    with card:
        st.markdown(f"### Segment {segment}")
        st.markdown(f"**{segment_labels[segment]}**")
        st.metric("Days", f"{count:,}")
        st.caption(f"{percentage:.1f}% of complete days")
        st.write(segment_descriptions[segment])
        st.info(recommendations[segment])

st.divider()

st.subheader("📈 Segment Distribution")

distribution = (
    df["Segment"]
    .value_counts()
    .sort_index()
    .rename(index=segment_labels)
)

st.bar_chart(distribution)

st.divider()

st.subheader("⚡ Average Daily Energy by Segment")

energy_by_segment = (
    df.groupby("Segment")["Daily_Energy_kWh"]
    .mean()
    .rename(index=segment_labels)
)

st.bar_chart(energy_by_segment)

st.divider()

st.subheader("🔥 Average Maximum Power by Segment")

max_power_by_segment = (
    df.groupby("Segment")["Max_Active_Power"]
    .mean()
    .rename(index=segment_labels)
)

st.bar_chart(max_power_by_segment)

st.divider()

st.subheader("📅 Monthly Segment Pattern")

monthly = (
    pd.crosstab(
        df["Month"],
        df["Segment"],
        normalize="index"
    ) * 100
)

monthly.columns = [
    segment_labels[int(column)]
    for column in monthly.columns
]

monthly.index = [
    pd.Timestamp(2000, month, 1).strftime("%b")
    for month in monthly.index
]

st.bar_chart(monthly)

st.caption(
    "Values represent the percentage of complete days belonging "
    "to each segment within each month."
)

st.divider()

st.subheader("🗓️ Weekday vs Weekend Pattern")

weekday_weekend = (
    pd.crosstab(
        df["IsWeekend"].map({
            0: "Weekday",
            1: "Weekend"
        }),
        df["Segment"],
        normalize="index"
    ) * 100
)

weekday_weekend.columns = [
    segment_labels[int(column)]
    for column in weekday_weekend.columns
]

st.bar_chart(weekday_weekend)

st.divider()

st.subheader("📅 Historical Day Explorer")

available_dates = (
    df["DateTime"]
    .dt.strftime("%Y-%m-%d")
    .tolist()
)

selected_date = st.selectbox(
    "Select a date",
    available_dates
)

selected_row = df[
    df["DateTime"].dt.strftime("%Y-%m-%d") == selected_date
].iloc[0]

segment = int(selected_row["Segment"])

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Daily Energy",
        f"{selected_row['Daily_Energy_kWh']:.2f} kWh"
    )

with col2:
    st.metric(
        "Maximum Power",
        f"{selected_row['Max_Active_Power']:.2f} kW"
    )

with col3:
    st.metric(
        "Peak / Average",
        f"{selected_row['Peak_to_Average_Ratio']:.2f}"
    )

with col4:
    st.metric(
        "Power Variability",
        f"{selected_row['Power_Variability']:.2f}"
    )

st.markdown(
    f"### Segment {segment}: {segment_labels[segment]}"
)

st.write(segment_descriptions[segment])
st.info(recommendations[segment])

st.divider()

st.subheader("🔮 Predict a New Daily Consumption Segment")

st.write(
    "Enter the nine behavioral features for a new complete day. "
    "The trained StandardScaler, PCA and KMeans models will assign "
    "the day to one of the four discovered segments."
)

col1, col2, col3 = st.columns(3)

with col1:
    daily_energy = st.number_input(
        "Daily Energy (kWh)",
        min_value=0.0,
        value=26.19,
        step=0.1
    )

    max_power = st.number_input(
        "Maximum Active Power (kW)",
        min_value=0.0,
        value=5.39,
        step=0.1
    )

    peak_average = st.number_input(
        "Peak-to-Average Ratio",
        min_value=0.0,
        value=5.21,
        step=0.1
    )

with col2:
    variability = st.number_input(
        "Power Variability",
        min_value=0.0,
        value=0.87,
        step=0.01
    )

    submeter_1 = st.number_input(
        "Submeter 1 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.11,
        step=0.01
    )

    submeter_2 = st.number_input(
        "Submeter 2 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.14,
        step=0.01
    )

with col3:
    submeter_3 = st.number_input(
        "Submeter 3 Ratio",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )

    reactive_active = st.number_input(
        "Reactive-to-Active Ratio",
        min_value=0.0,
        value=0.14,
        step=0.01
    )

    avg_voltage = st.number_input(
        "Average Voltage (V)",
        min_value=0.0,
        value=240.84,
        step=0.1
    )

st.divider()

if st.button("🔮 Predict Segment", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            daily_energy,
            max_power,
            peak_average,
            variability,
            submeter_1,
            submeter_2,
            submeter_3,
            reactive_active,
            avg_voltage
        ]],
        columns=cluster_features
    )

    scaled_input = scaler.transform(input_data)

    pca_input = pca.transform(scaled_input)

    predicted_segment = int(
        kmeans.predict(pca_input)[0]
    )

    predicted_label = segment_labels[predicted_segment]

    st.success(
        f"Predicted Segment {predicted_segment}: "
        f"{predicted_label}"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Segment",
            predicted_segment
        )

    with result_col2:
        st.metric(
            "Pattern",
            predicted_label
        )

    st.write(
        segment_descriptions[predicted_segment]
    )

    st.info(
        recommendations[predicted_segment]
    )

st.divider()

st.caption(
    "Model: StandardScaler + PCA + KMeans | "
    "Training data: 1,416 complete daily observations"
)
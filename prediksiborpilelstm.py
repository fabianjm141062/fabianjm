import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.title("LSTM Prediction of Bore Pile Parameters")
st.write(
    "Prediction of qa, sa, qh, yh, and bm using "
    "Long Short-Term Memory (LSTM)"
)

st.write("Prof. Dr. Fabian J. Manoppo – AI Data Analyst")

# ============================================================
# LOAD DATASET
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type="csv"
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ========================================================
    # DEFINE INPUT AND OUTPUT
    # ========================================================

    features = [
        'diameter',
        'length',
        'nspt1',
        'nspt2',
        'nspt3'
    ]

    targets = [
        'qa',
        'sa',
        'qh',
        'yh',
        'bm'
    ]

    required_columns = features + targets

    # Check columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.stop()

    # Remove missing data
    df = df[required_columns].dropna()

    X = df[features].values
    y = df[targets].values

    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    # ========================================================
    # RESHAPE FOR LSTM
    #
    # LSTM input:
    # samples × timesteps × features
    #
    # Here:
    # timestep = 1
    # features = 5
    # ========================================================

    X_train_lstm = X_train_scaled.reshape(
        X_train_scaled.shape[0],
        1,
        X_train_scaled.shape[1]
    )

    X_test_lstm = X_test_scaled.reshape(
        X_test_scaled.shape[0],
        1,
        X_test_scaled.shape[1]
    )

    # ========================================================
    # BUILD LSTM MODEL
    # ========================================================

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(1, len(features))
        )
    )

    model.add(Dropout(0.20))

    model.add(
        LSTM(
            units=32,
            return_sequences=False
        )
    )

    model.add(Dropout(0.20))

    model.add(
        Dense(
            units=32,
            activation='relu'
        )
    )

    # 5 output parameters
    model.add(
        Dense(
            units=len(targets),
            activation='linear'
        )
    )

    # ========================================================
    # COMPILE MODEL
    # ========================================================

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    st.subheader("LSTM Model Training")

    with st.spinner("Training LSTM model..."):

        history = model.fit(
            X_train_lstm,
            y_train_scaled,
            validation_split=0.20,
            epochs=300,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

    st.success("LSTM training completed.")

    # ========================================================
    # LOSS CURVE
    # ========================================================

    st.subheader("Training and Validation Loss")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        history.history['loss'],
        label='Training Loss'
    )

    ax.plot(
        history.history['val_loss'],
        label='Validation Loss'
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Learning Curve")

    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred_scaled = model.predict(
        X_test_lstm,
        verbose=0
    )

    # Return predictions to original scale
    y_pred = scaler_y.inverse_transform(
        y_pred_scaled
    )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    metric_results = []

    for i, target in enumerate(targets):

        mse = mean_squared_error(
            y_test[:, i],
            y_pred[:, i]
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            y_test[:, i],
            y_pred[:, i]
        )

        r2 = r2_score(
            y_test[:, i],
            y_pred[:, i]
        )

        metric_results.append({
            "Target": target,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        })

    metric_df = pd.DataFrame(metric_results)

    st.subheader("LSTM Model Performance")

    st.dataframe(metric_df)

    # ========================================================
    # ACTUAL VS PREDICTED
    # ========================================================

    for i, target in enumerate(targets):

        st.subheader(
            f"Actual vs Predicted – {target}"
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        actual = y_test[:, i]
        predicted = y_pred[:, i]

        ax.scatter(
            actual,
            predicted
        )

        min_val = min(
            actual.min(),
            predicted.min()
        )

        max_val = max(
            actual.max(),
            predicted.max()
        )

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle='--',
            label='Perfect Prediction'
        )

        ax.set_xlabel(
            f"Actual {target}"
        )

        ax.set_ylabel(
            f"Predicted {target}"
        )

        ax.set_title(
            f"LSTM Prediction – {target}"
        )

        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    # ========================================================
    # ACTUAL VS PREDICTED TABLE
    # ========================================================

    results = pd.DataFrame()

    for i, target in enumerate(targets):

        results[
            f"Actual_{target}"
        ] = y_test[:, i]

        results[
            f"Predicted_{target}"
        ] = y_pred[:, i]

        results[
            f"Error_{target}"
        ] = (
            y_test[:, i]
            - y_pred[:, i]
        )

    st.subheader(
        "Actual vs Predicted Results"
    )

    st.dataframe(results)

    # ========================================================
    # MANUAL PREDICTION
    # ========================================================

    st.subheader(
        "Bore Pile Parameter Prediction"
    )

    diameter = st.number_input(
        "Pile Diameter",
        min_value=0.0
    )

    length = st.number_input(
        "Pile Length",
        min_value=0.0
    )

    nspt1 = st.number_input(
        "N-SPT 1",
        min_value=0.0
    )

    nspt2 = st.number_input(
        "N-SPT 2",
        min_value=0.0
    )

    nspt3 = st.number_input(
        "N-SPT 3",
        min_value=0.0
    )

    if st.button(
        "Predict Bore Pile Parameters"
    ):

        new_data = np.array([[
            diameter,
            length,
            nspt1,
            nspt2,
            nspt3
        ]])

        new_scaled = scaler_X.transform(
            new_data
        )

        new_lstm = new_scaled.reshape(
            1,
            1,
            len(features)
        )

        prediction_scaled = model.predict(
            new_lstm,
            verbose=0
        )

        prediction = scaler_y.inverse_transform(
            prediction_scaled
        )

        prediction_df = pd.DataFrame(
            prediction,
            columns=targets
        )

        st.subheader(
            "LSTM Prediction Results"
        )

        st.dataframe(
            prediction_df
        )

    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    csv = results.to_csv(
        index=False
    ).encode('utf-8')

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="LSTM_BorePile_Predictions.csv",
        mime="text/csv"
    )

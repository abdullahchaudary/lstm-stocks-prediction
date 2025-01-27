# Stock Prediction Application

## Overview
This application is designed to predict stock trends using a Long Short-Term Memory (LSTM) neural network. It supports multi-stock time series predictions and provides insights such as performance metrics, market trends (bull vs bear), and visualizations of predictions.

---

## Goals of the Application

- **Stock Trend Prediction:** Predict future stock prices based on historical data.
- **Market Analysis:** Analyze bull and bear market trends.
- **Visualization:** Provide interactive and comprehensive visual insights into stock performance.
- **Performance Metrics:** Calculate and display metrics like MSE, MAE, R², Precision, Recall, and F1 Score for each stock.

---

## Features

1. **Interactive File Upload:** Upload a CSV file with stock data.
2. **Data Cleaning:** Handle missing values using interpolation and forward/backward filling.
3. **LSTM-based Prediction:** Train and test a neural network on the uploaded data.
4. **Visual Analysis:** Generate charts for:
   - Training loss
   - Testing results
   - Bull vs Bear market trends
   - Performance metrics
5. **Comprehensive Metrics:** Evaluate predictions using:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R² Score
   - Precision
   - Recall
   - F1 Score

---

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/abdullahchaudary/lstm-stocks-prediction.git
   cd lstm-stocks-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.streamlit.py
   ```

---

## CSV Input Format
The application requires a CSV file with the following columns:

1. **INDEX Date:** The dates of the index-level data.
2. **INDEX Price:** The price values of the index-level data.
3. **Additional Stock Data:**
   - Each stock must have its own pair of columns:
     - A `Date` column.
     - A `Price` column.

### Example Input File
| INDEX Date  | INDEX Price | StockA Date | StockA Price | StockB Date | StockB Price |
|-------------|-------------|-------------|--------------|-------------|--------------|
| 2023-01-01  | 100.5       | 2023-01-01  | 50.2         | 2023-01-01  | 75.8         |
| 2023-01-02  | 101.0       | 2023-01-02  | 50.4         | 2023-01-02  | 75.9         |

---

## Sample Data and Output

1. **Sample Input File:** [Download CSV](https://github.com/abdullahchaudary/lstm-stocks-prediction/blob/main/sample_index.csv)
2. **Sample PDF Output:** [View PDF](https://github.com/abdullahchaudary/lstm-stocks-prediction/blob/main/sample_output.pdf)

---

## Inner Workings

### LSTM Model
The LSTM model is designed as follows:

1. **Architecture:**
   - Input Layer: Accepts scaled stock prices.
   - LSTM Layers: Two LSTM layers to capture temporal dependencies.
   - Fully Connected Layer: Outputs the predicted stock price.

2. **Training:**
   - The model is trained using the Mean Squared Error (MSE) loss function.
   - Optimizer: Adam optimizer is used to minimize the loss function.
   - Data Batching: Training data is split into batches of 64 for efficient learning.

3. **Testing:**
   - After training, the model predicts stock prices on the test dataset.
   - The predicted values are compared against actual values to calculate metrics.

### Data Handling

- **Preprocessing:**
  - Missing values are handled using:
    - Linear interpolation.
    - Backward fill (bfill).
    - Forward fill (ffill).
  - Dates are converted to datetime format and aligned across stocks.

- **Scaling:**
  - Prices are scaled between 0 and 1 using `MinMaxScaler` for optimal model performance.

- **Sequence Creation:**
  - Data is split into sequences of 50 days to capture temporal dependencies.

### Visualizations

1. **Training Loss:**
   - Plots the loss over epochs to show how well the model is learning.

2. **Testing Results:**
   - Plots the actual vs predicted prices.

3. **Bull vs Bear Market Analysis:**
   - Highlights average predictions during bull and bear markets.

4. **Performance Metrics:**
   - Bar charts for each metric (MSE, MAE, R², etc.) across stocks.

---

## Contributions
Developed in collaboration with [Abdullah Chaudary](https://abchaudary.me). See the full project on [GitHub](https://github.com/abdullahchaudary/lstm-stocks-prediction).

---

## Contact
For issues or contributions, please visit the [GitHub repository](https://github.com/abdullahchaudary/lstm-stocks-prediction).

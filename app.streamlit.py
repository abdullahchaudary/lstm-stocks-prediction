import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
from datetime import datetime

# Header with credits and links
st.markdown(
    """<hp style='text-align: left;'>
    Developed by <a href='https://abchaudary.me' target='_blank'>Abdullah Chaudary</a>
    | <a href='https://github.com/abdullahchaudary/lstm-stocks-prediction' target='_blank'>See project on GitHub</a>
    </hp>""",
    unsafe_allow_html=True
)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length, :-1]
        y = self.data[idx + self.sequence_length, -1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Streamlit app
st.title("Multistock Time Series Prediction with LSTM")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("#### Dataset Preview")
    st.dataframe(df.head())

    # Identify `INDEX Date` and `INDEX Price` columns
    index_date_col = 'INDEX Date'
    index_price_col = 'INDEX Price'

    if index_date_col not in df.columns or index_price_col not in df.columns:
        st.error("The uploaded file must contain 'INDEX Date' and 'INDEX Price' columns.")
        st.stop()

    # Process index-level data
    df[index_date_col] = pd.to_datetime(df[index_date_col], errors='coerce')
    if df[index_date_col].isnull().any():
        st.error("Some dates in 'INDEX Date' could not be parsed. Please clean your data.")
        st.stop()

    df = df.sort_values(by=index_date_col).reset_index(drop=True)
    df = df.set_index(index_date_col).asfreq('D').reset_index()

    # Handle missing index prices
    df[index_price_col] = df[index_price_col].replace({',': ''}, regex=True).astype(float)
    df[index_price_col] = df[index_price_col].interpolate(method='linear')
    df[index_price_col] = df[index_price_col].bfill()
    df[index_price_col] = df[index_price_col].ffill()

    # Process each stock
    stock_columns = [col for col in df.columns if col not in [index_date_col, index_price_col]]
    stock_pairs = [(stock_columns[i], stock_columns[i + 1]) for i in range(0, len(stock_columns), 2)]

    all_results = []  # Store results for index-level analysis

    for date_col, price_col in stock_pairs:
        st.write(f"#### Processing Stock: {price_col}")

        try:
            # Process stock data
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[price_col] = df[price_col].replace({',': ''}, regex=True).astype(float)

            # Align stock dates with index dates
            stock_df = df[[index_date_col, index_price_col, date_col, price_col]].dropna()
            stock_df = stock_df.rename(columns={date_col: 'Date', price_col: 'Price'})
            stock_df = stock_df[['Date', 'Price']].set_index('Date').asfreq('D').reset_index()

            # Handle missing stock prices
            stock_df['Price'] = stock_df['Price'].interpolate(method='linear')
            stock_df['Price'] = stock_df['Price'].bfill()
            stock_df['Price'] = stock_df['Price'].ffill()

            close_prices = stock_df['Price'].values.reshape(-1, 1)

            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(close_prices)

            sequence_length = 50
            dataset = np.hstack([scaled_data, scaled_data])
            train_size = int(len(dataset) * 0.7)
            train_data = dataset[:train_size]
            test_data = dataset[train_size - sequence_length:]

            train_dataset = TimeSeriesDataset(train_data, sequence_length)
            test_dataset = TimeSeriesDataset(test_data, sequence_length)

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            st.write(f"Train and test datasets prepared for {price_col}.")

            # Model setup
            input_size = 1
            hidden_size = 50
            num_layers = 2
            output_size = 1

            model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Training
            st.write(f"#### Training the Model for {price_col}")
            num_epochs = 20
            model.train()
            train_losses = []

            # Create a scrollable container for epoch updates
            with st.container():
                epoch_log = st.empty()  # Placeholder for scrollable logs
                log_lines = []  # Store all logs

                for epoch in range(num_epochs):
                    epoch_loss = 0
                    for x_batch, y_batch in train_loader:
                        x_batch = x_batch.view(x_batch.size(0), x_batch.size(1), -1)  # Reshape to [batch_size, sequence_length, features]
                        optimizer.zero_grad()
                        outputs = model(x_batch)  # Ensure 3D input
                        loss = criterion(outputs, y_batch.unsqueeze(-1))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    train_losses.append(epoch_loss / len(train_loader))

                    # Update logs
                    log_lines.append(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
                    epoch_log.text("\n".join(log_lines))  # Show all logs with scrollable behavior

            # Plot training loss
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss for {price_col}')
            plt.legend()
            st.pyplot(plt)

            # Testing
            st.write(f"#### Testing the Model for {price_col}")
            model.eval()
            predictions, actuals = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.view(x_batch.size(0), x_batch.size(1), -1)  # Reshape to [batch_size, sequence_length, features]
                    outputs = model(x_batch)  # Ensure 3D input
                    predictions.append(outputs.numpy())
                    actuals.append(y_batch.numpy())

            predictions = np.concatenate(predictions).flatten()
            actuals = np.concatenate(actuals).flatten()

            # Inverse scaling
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

            # Compute metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            precision = np.nan if len(set(actuals)) < 2 else precision_score((actuals > np.mean(actuals)).astype(int), (predictions > np.mean(actuals)).astype(int))
            recall = np.nan if len(set(actuals)) < 2 else recall_score((actuals > np.mean(actuals)).astype(int), (predictions > np.mean(actuals)).astype(int))
            f1 = np.nan if len(set(actuals)) < 2 else f1_score((actuals > np.mean(actuals)).astype(int), (predictions > np.mean(actuals)).astype(int))

            st.write(f"#### Performance Metrics for {price_col}")
            st.write(f"Mean Squared Error (MSE): {mse:.4f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
            st.write(f"R-Squared (R2): {r2:.4f}")
            st.write(f"Precision: {precision if not np.isnan(precision) else 'N/A'}")
            st.write(f"Recall: {recall if not np.isnan(recall) else 'N/A'}")
            st.write(f"F1 Score: {f1 if not np.isnan(f1) else 'N/A'}")

            # Plot testing results
            plt.figure(figsize=(10, 6))
            plt.plot(actuals, label='Actual Prices', color='blue')
            plt.plot(predictions, label='Predicted Prices', color='red')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.title(f'Testing Results for {price_col}')
            plt.legend()
            st.pyplot(plt)

            # Market trend analysis
            st.write(f"#### Market Trend Analysis for {price_col}")
            bull_market = predictions[actuals > np.mean(actuals)]
            bear_market = predictions[actuals < np.mean(actuals)]
            st.write(f"Bull Market Average Prediction: {np.mean(bull_market):.4f}")
            st.write(f"Bear Market Average Prediction: {np.mean(bear_market):.4f}")

            # Plot market trends
            plt.figure(figsize=(10, 6))
            plt.hist(bull_market, bins=20, alpha=0.7, label='Bull Market', color='green')
            plt.hist(bear_market, bins=20, alpha=0.7, label='Bear Market', color='red')
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            plt.title(f'Market Trends for {price_col}')
            plt.legend()
            st.pyplot(plt)

            all_results.append({
                'Stock': price_col,
                'Bull Market Avg': np.mean(bull_market),
                'Bear Market Avg': np.mean(bear_market),
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })

        except Exception as e:
            st.error(f"Error processing stock {price_col}: {e}")

    # Index-level detailed analysis
    st.write("### Index-Level Analysis")
    if all_results:
        results_df = pd.DataFrame(all_results)
        st.write("#### Summary of Predictions Across All Stocks")
        st.dataframe(results_df)

        # Visualize comparisons
        st.write("#### Bull vs Bear Market Averages by Stock")
        plt.figure(figsize=(10, 6))
        x = np.arange(len(results_df))
        width = 0.35
        plt.bar(x - width/2, results_df['Bull Market Avg'], width, label='Bull Market')
        plt.bar(x + width/2, results_df['Bear Market Avg'], width, label='Bear Market')
        plt.xticks(x, results_df['Stock'], rotation=45)
        plt.ylabel('Average Predictions')
        plt.title('Market Analysis Across Stocks')
        plt.legend()
        st.pyplot(plt)

        # Display performance metrics summary
        st.write("#### Performance Metrics Across All Stocks")
        performance_metrics = results_df[['Stock', 'MSE', 'MAE', 'R2', 'Precision', 'Recall', 'F1 Score']]
        st.dataframe(performance_metrics)

        # Plot performance metrics
        st.write("#### Performance Metrics Visualized")
        metrics = ['MSE', 'MAE', 'R2', 'Precision', 'Recall', 'F1 Score']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(results_df['Stock'], results_df[metric], color='skyblue')
            plt.xlabel('Stock')
            plt.ylabel(metric)
            plt.title(f'{metric} Across Stocks')
            plt.xticks(rotation=45)
            st.pyplot(plt)

    else:
        st.write("No results to analyze.")

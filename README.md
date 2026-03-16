# F1Oracle: Formula 1 Race Predictor

An end-to-end Machine Learning pipeline built with PyTorch and `fastf1` that predicts Formula 1 race results. The model learns from historical race data and uses live Qualifying results to forecast the final grid order.


 Project Overview

This project is broken down into three main Jupyter Notebooks, representing a complete data science workflow: from data scraping and preprocessing, to deep learning, and finally, live inference.

 1. Data Collection (`01_Data_Collection.ipynb`)
- Automates the extraction of historical F1 race data using the `fastf1` API.
- Pulls crucial telemetry and session data: Grid Position, Finish Position, Team, and Track.
- Dynamically maps constructors to their engine suppliers (e.g., Haas -> Ferrari, Sauber -> Audi).
- Cleans and formats the data into a standardized CSV for training.

 2. Model Training (`02_Model_Training.ipynb`)
- Loads the historical dataset and prepares it using Scikit-learn's `StandardScaler` and `LabelEncoder`.
- Defines **F1Oracle**: A custom Multi-Layer Perceptron (MLP) built from scratch in PyTorch.
- Trains the model to understand the complex relationships between starting grid position, engine power, and track layout.
- Saves the trained model weights (`.pth`) and encoder states (`.joblib`) for future inference.

 3. Race Simulator (`03_Race_Simulator.ipynb`)
- The live inference engine for race weekends.
- Connects to `fastf1` to scrape the **live Qualifying results** of an upcoming Grand Prix.
- Automatically processes the fresh grid, handles new/unseen variables dynamically, and feeds the data into the trained PyTorch model.
- Outputs a sorted, tie-broken prediction of the final race standings.

 Tech Stack

- Deep Learning: PyTorch
- Data Engineering: Pandas, NumPy, Scikit-learn
- F1 Telemetry: FastF1 API
- Environment: Jupyter Notebook / Google Colab


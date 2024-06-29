**Stock Market Prediction with Guided Stochastic Gradient Descent (GSGD)**
<br>This project explores the application of Guided Stochastic Gradient Descent (GSGD) with Recurrent Neural Networks (RNNs) to enhance the performance of stock price prediction models. The study compares the performance of GSGD-optimized RNNs (GRNN) with standard RNN and Long Short-Term Memory (LSTM) networks.
<br>
<br>**Overview**
<br>Stock market prediction is a challenging task due to the complex and volatile nature of financial markets. This project uses machine learning techniques, particularly RNNs, to model sequential data for predicting stock price movements. The project investigates the use of GSGD, an advanced optimization technique, to improve the performance of RNNs in handling inconsistent data and revisiting batches based on loss.
<br>
<br>**Installation**
<br> &nbsp;Clone the repository:
<br> &nbsp; &nbsp;git clone https://github.com/your-username/stock-market-prediction.git
<br> &nbsp; &nbsp;cd stock-market-prediction
<br> &nbsp;Create a virtual environment:
<br> &nbsp; &nbsp;python3 -m venv venv
<br> &nbsp; &nbsp;source venv/bin/activate  # On Windows use `venv\Scripts\activate`
<br> &nbsp;Install dependencies:
<br> &nbsp; &nbsp;pip install -r requirements.txt
<br>
<br>**Usage**
<br> &nbsp;Prepare the data:
<br> &nbsp; &nbsp;Ensure your dataset is placed in the data/ directory.
<br> &nbsp; &nbsp;Modify data_preparation.py to point to your dataset and preprocess it as needed.
<br>Train the models:
<br> &nbsp; &nbsp;Run the main script to train and evaluate the models:
<br> &nbsp; &nbsp;python main.py
<br>
<br>**Training and Evaluation**
<br>The project includes training scripts for three types of models:
<br> &nbsp;Standard RNN (train_rnn.py)
<br> &nbsp;LSTM (train_lstm.py)
<br> &nbsp;GRNN with GSGD (train_gsgd.py)
<br>The main.py script handles the training and evaluation of these models, including the calculation of metrics such as accuracy, F1 score, and ROC-AUC.
<br>
<br>**Results**
<br>The results of the study are summarized in the results/ directory. Key metrics include accuracy, F1 score, and ROC-AUC for each model.

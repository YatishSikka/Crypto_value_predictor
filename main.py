# Importing all the necessary libraries
import numpy as n
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as p
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Importing data and rectifying loss of data by filling the NaN values with mean of the whole field
data = p.read_csv('LTC-USD.csv', date_parser=True)
count = 0
for j in data['Open']:
  if(math.isnan(j)):
    count+=1
data['Open'].fillna(value=data['Open'].mean(), inplace=True)
data['High'].fillna(value=data['High'].mean(), inplace=True)
data['Low'].fillna(value=data['Low'].mean(), inplace=True)
data['Close'].fillna(value=data['Close'].mean(), inplace=True)
data['Adj Close'].fillna(value=data['Adj Close'].mean(), inplace=True)
data['Volume'].fillna(value=data['Volume'].mean(), inplace=True)
print(count)

# Splitting data into traing and test data and dropping all the unnecessay columns from the table
training_data = data[data['Date'] <= '2021-10-31'].copy()
test_data = data[data['Date'] > '2021-10-31'].copy()

training_data = training_data.drop(['Date', 'Adj Close'], axis=1)

# Scaling the data to fit a particular range
scaler = MinMaxScaler()
train_data = scaler.fit_transform(training_data)
X_train = []
Y_train = []

for i in range(60, train_data.shape[0]):
    X_train.append(train_data[i - 60:i])
    Y_train.append(train_data[i, 0])

X_train = n.array(X_train)
Y_train = n.array(Y_train)

# Creating multiple layers of neural network
model = Sequential()
model.add(LSTM(units = 50, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
model.add(Dropout(0.2))
model.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.4))
model.add(LSTM(units = 120, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(units =1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(X_train, Y_train, epochs=40, batch_size=25, validation_split=0.1)

# Calculating the loss and validation loss and plotting a graph to compare them
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Creating a test data to predict the trend of the market before 60 days
part_60_days = training_data.tail(60)
df = part_60_days.append(test_data, ignore_index=True)
df = df.drop(['Date', 'Adj Close'], axis=1)
df.head()
inputs = scaler.transform(df)

X_test = []
Y_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    Y_test.append(inputs[i, 0])

X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Scaling the data for plotting the final graph
Y_pred = model.predict(X_test)

scale = 1/scaler.scale_[0]
Y_test = Y_test*scale
Y_pred = Y_pred*scale

# Plotting the final graph to predict the trend of the market
plt.figure(figsize=(14, 5))
plt.plot(Y_test, color='red', label='Real LTC Price')
plt.plot(Y_pred, color='green', label='Predicted BTC Price')
plt.title('LTC Price Prediction using RNN-LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

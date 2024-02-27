# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

This dataset presents a captivating challenge due to the intricate relationship between the input and output columns. The complex nature of this connection suggests that there may be underlying patterns or hidden factors that are not readily apparent.

## Neural Network Model

![image](https://github.com/Yamunaasri/basic-nn-model/assets/115707860/29d38b6e-9112-4ebd-ab74-f3631b0fb31c)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```
Name: T S Yamunaasri
Register Number: 212222240117
```
### DEPENDENCIES
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```
### DATA FROM GSHEETS
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd


auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('dl_data_exp1').sheet1


rows = worksheet.get_all_values()
```
### DATA PROCESSING
```
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()

df['Input']=pd.to_numeric(df['Input'])
df['Output']=pd.to_numeric(df['Output'])

X = df[['Input']].values
y = df[['Output']].values

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()
Scaler.fit(x_train)

x_train.shape
x_train1 = Scaler.transform(x_train)
x_train1.shape
```
### MODEL ARCHITECTURE AND TRAINING
```
model = Sequential([
    Dense(units = 5,activation = 'relu',input_shape=[1]),
    Dense(units = 2, activation = 'relu'),
    Dense(units = 1)
])

model.compile(optimizer='rmsprop', loss = 'mae')

model.fit(x_train1,y_train,epochs = 2000)

model.summary()
```
### LOSS CALCULATION
```
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
```
### PREDICTION
```
x_test1 = Scaler.transform(x_test)
model.evaluate(x_test1,y_test)

x_n = [[21]]
x_n1 = Scaler.transform(x_n)
model.predict(x_n1)

```
## Dataset Information

![image](https://github.com/Yamunaasri/basic-nn-model/assets/115707860/df6b6339-269b-4c5e-a988-62d9e27ba591)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Yamunaasri/basic-nn-model/assets/115707860/76cf4b8f-7e8f-406f-8fc4-e9639c1f72cb)

### Test Data Root Mean Squared Error

![image](https://github.com/Yamunaasri/basic-nn-model/assets/115707860/b8aee8e1-b2ef-4f3f-8b5e-48f4ddadb3a4)

### New Sample Data Prediction

![image](https://github.com/Yamunaasri/basic-nn-model/assets/115707860/3f19bca2-7d21-4de0-acff-45893df9ab66)

## RESULT

Summarize the overall performance of the model based on the evaluation metrics obtained from testing data as a regressive neural network based prediction has been obtained.

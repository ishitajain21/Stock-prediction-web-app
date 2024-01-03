import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st 

yf.pdr_override()

start = '2012-01-01'
# because the use of Deep learning really hit off in 2012
end = '2023-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')


df = pdr.get_data_yahoo(user_input,start,end)

#Describing Data 
st.subheader('Data from 2010 - 2019')
st.write(df.describe())
# will later be dynamic 

#Visualizations 
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)
#splitting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])
from sklearn.preprocessing import MinMaxScaler
scalar =  MinMaxScaler(feature_range=(0,1))
data_training_array = scalar.fit_transform(data_training)
#load keras model 
model = load_model('keras_model.h5')

# testing part 
past_100_days= data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing],ignore_index = True)
input_data = scalar.fit_transform(final_df)



X_test = []
y_test = []
for i in range(100,input_data.shape[0]): 
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
X_test,y_test = np.array(X_test),np.array(y_test)

# making predictions 

y_predicted = model.predict(X_test)
scale_factor = 1/scalar.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# final graph 
st.subheader('predictions vs original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original price')
plt.plot(y_predicted,'r',label = 'Predicted price')
plt.xlabel('Time(days)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
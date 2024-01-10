import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas_datareader import data as pdr
import yfinance as yf
from keras.models import load_model
import streamlit as st 
import datetime
from dateutil.relativedelta import relativedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

yf.pdr_override()

today = datetime.date.today()

start = today - relativedelta(years=5)
end = today

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker','AAPL')


df = pdr.get_data_yahoo(user_input,start,end)
df = df.reset_index()
date = df['Date']
df = df[['Date','Close']]

df['day_of_week'] = df['Date'].dt.dayofweek 


#Visualizations 
st.subheader('Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.legend()
st.pyplot(fig)
#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7): int(len(df))])

scalar =  MinMaxScaler(feature_range=(0,1))
data_training_array = scalar.fit_transform(data_training)
#load keras model 
model = load_model('keras_model.h5')

# testing part 
past_100_days= data_training.tail(100)
# the 'feautures' to use to predict the values 



final_df = pd.concat([past_100_days, data_testing],ignore_index = True)
#null_df = [None, None, None, None, None]
#final_df = pd.concat([final_df,np.array(null_df)], ignore_index=True)
input_data = scalar.fit_transform(final_df)



X_test = []
y_test = []
# start from day 100 and go till the end of the data
for i in range(100,input_data.shape[0]): 
    X_test.append(input_data[i-100:i])
    # X_test will consist of the array that will be the past 100 days for i's date 
    y_test.append(input_data[i,0])
    # y_test will have the value of the close price of ith day 


X_test,y_test = np.array(X_test),np.array(y_test)

# making predictions 

y_predicted = model.predict(X_test)
scale_factor = 1/scalar.scale_
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

x =user_input +  ' closing price predictions vs original in the past'

# date_float = date.values.astype("float64")
# y_test_with_date = np.append(date_float,y_test,axis=1)
# y_predicted_with_date = np.append(date_float,y_predicted,axis=1)

st.subheader(x)
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original price')
plt.plot(y_predicted,'r',label = 'Predicted price')
plt.xlabel('Time(days)')
plt.ylabel('Price($)')
plt.legend()
st.pyplot(fig2)

# cvscores = []
# scores = model.evaluate(X_test, y_test/scale_factor, verbose=0)
# x = "%s: %.2f%%" % (model.metrics_names[1], scores[1]*100)

# st.subheader(x)
# 3 tabs for whether you are looking to buy, sell, hold 
y_test = y_test/scale_factor
for i in range(1,6):
    past_100_days = y_test[-100:]    
    past_100_days = past_100_days.reshape(1,100,1)  
    day = model.predict(past_100_days)
    print(day.shape)
    day = day.reshape(1,1)
    y_test = np.append(y_test, day)
y_test = y_test * scale_factor
st.header('Price estimate for next days:')
count  = -5
for i in range(0,5):
    count = count + i
    x= str(i+1) + 'th from today will be: $' +  str(round(y_test[count],2) )
    st.subheader(x)
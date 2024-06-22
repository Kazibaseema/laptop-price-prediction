import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Predictor")

# ask user which brand laptop do you want
company = st.selectbox('Brand', data['Company'].unique())

# type of laptop
type = st.selectbox('Type', data['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight of the laptop
weight = st.number_input('Weight of the laptop')

# touchScreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

#IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# ScreenSize
screen_size = st.number_input('Screen_Size')

# Resolution
resolution = st.selectbox('Screen Resolution',
                         ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2560x1600', '2560x1440',
                          '2304x1440'])

# CPU
cpu = st.selectbox('CPU', data['Cpu brand'].unique())

#
hdd = st.selectbox('HDD(in GB)', [8, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1824])

gpu = st.selectbox('GPU', data['Gpu brand'].unique())

# os
os = st.selectbox('OS', data['os'].unique())

if st.button('Predict Price'):
    #Query

    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

#here the x_res is string so we have to pass it as int
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)
    st.title("the predicted price of this configuration is " + str(int(pipe.predict(query[0]))))

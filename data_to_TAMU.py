import Adafruit_ADXL345
import requests
import numpy as np
import time
accel = Adafruit_ADXL345.ADXL345()
accel.set_data_rate(Adafruit_ADXL345.ADXL345_DATARATE_800_HZ)
NUM_OF_SEC_TO_RUN = 10
URL = 'http://www.google.com'

def get_data():
   x = []
   Y = []
   Z = []
   t = []
   start_time=time.time()
   while time.time()<=start_time+NUM_OF_SEC_TO_RUN:
       xx, yx, zx = accel.read()
       x.append(xx)
       Y.append(yx)
       Z.append(zx)
       t.append(time.time())


   t = t[6000:]
   x = x[6000:]
   Y = Y[6000:]
   Z = Z[6000:]
   N = len(t)#length of the array
   Fs = 1/(t[6002]-t[6001]) #sample rate (Hz)
   T = 1/Fs;
   print("# Samples:",N,Fs,T)
   return x,Y,Z,t,N

def insert_data(X,T):
   data = {'Amplitude':X,'Timestamp':T}
   #print(data)
   requests.post(url=URL,json=data)
   
	
while(True):
  X,Y,Z,T,limit = get_data()
  insert_data(X,T)
  print("done") 
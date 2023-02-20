from scipy.signal import stft
import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#function def

#muti-channel batch STFT
def batch_stft(x):
  siz = x.shape
  xf = np.zeros((siz[0],65,75,6))
  for i in range(siz[0]):
    for j in range(6):
      tem = x[i,:,j]
      f, t, Zxx = stft(tem, window='hann', nperseg=128, noverlap=94)
      maxx = np.max(abs(Zxx))
      xf[i,:,:,j] = abs(Zxx)/0.85

  return xf

#muti-channel batch CWT
def batch_cwt(x):
  fs = 250
  dt = 1/fs
  f_min = 5
  f_max = 60
  scarle = 80
  siz = x.shape
  xf = np.zeros((siz[0],scarle,siz[1],siz[2]))
  for i in range(siz[0]):
    for j in range(siz[2]):
      tem = x[i,:,j]
      scalogram = cwt(tem, dt, 8, f_min, f_max, nf=scarle)
      x_cwt = np.abs(scalogram)
      xf[i,:,:,j] = x_cwt

  return xf

def resultshow(classes,label):
  lis=np.zeros(classes.shape[0])
  for i in range(classes.shape[0]):
   jeg = np.where(classes[i,:] == max(classes[i,:]))
   lis[i] = jeg[0]

  tru=np.zeros(classes.shape[0])
  for i in range(classes.shape[0]):
    jeg2 = np.where(label[i,:] == max(label[i,:]))
    tru[i] = jeg2[0]
  
  target = ['earthquake','quake','rockfall','enviroemnt noise']
  print(classification_report(tru, lis, target_names=target))
  print(confusion_matrix(tru, lis))

import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import torch 
import tensorflow as tf 
import sklearn
from sklearn.metrics import classification_report,f1_score,confusion_matrix,accuracy_score,precision_score,recall_score

#device configuration
device= torch.device('cuda' if torch.cuda.is_available() else'cpu')
print(f'using device:',device)

#dataset directories 
test_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\test'
train_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\train'
val_split=r'C:\Users\Hp\Documents\dataset folders\skin disease dataset\6GB HAM10000 DATA\output_folder\val'


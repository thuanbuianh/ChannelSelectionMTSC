import sys 
import os
sys.path.insert(0, os.getcwd())
sys.path.append(".")
sys.path.append("..")

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import time
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe

from src.kmeans import kmeans
from  src.classelbow import ElbowPair # ECP
from src.elbow import elbow # ECS..

# selects class prototype
center='mad' # options: mean, median
#elb = kmeans()
elb  = elbow(distance = 'eu', center=center) # Select elbow class sum
#elb = ElbowPair(distance = 'eu', center=center) # Selects elbow class Pair
#elb = None

train = "data/ArticularyWordRecognition/ArticularyWordRecognition_TRAIN.ts"
test = "data/ArticularyWordRecognition/ArticularyWordRecognition_TEST.ts"

train_x, train_y = load_from_tsfile_to_dataframe(train)
test_x, test_y = load_from_tsfile_to_dataframe(test)
print(train_x.shape)
elb.fit(train_x, train_y)
df = elb.transform(train_x)
print(df.shape)
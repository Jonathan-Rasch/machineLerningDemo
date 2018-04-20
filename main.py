import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataGen
########################################################################################################################
# getting data
########################################################################################################################
test,train = dataGen.getData(10000,0.3)
print(test.head())
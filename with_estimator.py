import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataGen
import random
import logging
import math
import inspect_checkpoint
import time
########################################################################################################################
# PARAMETERS
########################################################################################################################
logging.getLogger().setLevel(logging.INFO)
RANDOM_STATE = 101
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
########################################################################################################################
# Supporting functions
########################################################################################################################
def evaluate(model):
    prediction_lst = model.predict(input_fn=infn_pred)
    predictions = []
    for pred in prediction_lst:
        value_raw = pred['predictions']
        value = y_scalar.inverse_transform(value_raw.reshape(1,-1))
        predictions.append(value)
    #computing average error
    error_sum = 0
    for index,actual_value in enumerate(y_scalar.inverse_transform(df_test[['delivery_time_min']].values)):
        error = math.fabs(predictions[index] - actual_value)
        error_sum += error
    avg_err = error_sum/len(predictions)
    print("ERROR: " + str(avg_err) + " minutes")
    return avg_err

########################################################################################################################
# getting data
########################################################################################################################
df_train,df_test,x_scalar,y_scalar,scatterMatrix = dataGen.getData(1000,0.3,visualise=True)
########################################################################################################################
# generating graph
########################################################################################################################
figure = plt.figure(2)
axis_error = figure.add_subplot(111)
axis_error.set_autoscaley_on(True)
#axis_error.set_ylim([0,60])
axis_error.set_yticks(list(range(0,500,5)),minor=False)
axis_error.set_title("Avg model error (in minutes) vs training iterations ")
plt.xlabel("number of training iterations")
plt.ylabel("error (minutes) in predicted delivery time vs actual delivery time")
plt.figure(1)
for axis in scatterMatrix.flatten():
    axis.plot()
plt.show(block=False)
########################################################################################################################
# constructing feature columns
########################################################################################################################
feature_columns = ['distance_km','order_size','vehicle_type','hour_of_day']
f_distance = tf.feature_column.numeric_column(key='distance_km')
f_orderSize = tf.feature_column.numeric_column(key='order_size')
f_vehicle = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(key='vehicle_type',vocabulary_list=['none','car','scooter','bicicle']),dimension=2)
f_hour = tf.feature_column.numeric_column(key='hour_of_day')
f_cols = [f_distance,f_orderSize,f_vehicle,f_hour]
########################################################################################################################
# constructing input functions
########################################################################################################################
infn_train1 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['delivery_time_min']],num_epochs=500,batch_size=5,shuffle=True)
infn_train2 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['delivery_time_min']],num_epochs=500,batch_size=10,shuffle=True)
infn_train3 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['delivery_time_min']],num_epochs=500,batch_size=50,shuffle=True)
infn_lateStage = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['delivery_time_min']],num_epochs=500,batch_size=128,shuffle=True)
TRAINING_FUNCT = infn_train1
infn_test = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],y=df_test[['delivery_time_min']],num_epochs=1,shuffle=False)
infn_pred = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],num_epochs=1,shuffle=False)
########################################################################################################################
# constructing model
########################################################################################################################
steps = 1
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 5,  # Save checkpoints every 30 seconds.
    keep_checkpoint_max = 2,       # Retain the 10 most recent checkpoints.
    model_dir='pizza'
)
model = tf.estimator.DNNRegressor(hidden_units=[16,16,8],feature_columns=f_cols,dropout=0.1,activation_fn=tf.nn.elu,config=my_checkpointing_config)
########################################################################################################################
# training
########################################################################################################################
model.train(input_fn=TRAINING_FUNCT, steps=1)

error = evaluate(model)
errors = []
x_axis = []
index = 0
while(error>5.5):
    model.train(input_fn=TRAINING_FUNCT,steps=steps)
    #weights = model.get_variable_value("dnn/hiddenlayer_0/kernel/part_0")
    error = evaluate(model)
    if(error>60):
        continue
    index += steps
    x_axis.append(index)
    errors.append(error)
    line, = axis_error.plot(x_axis,errors,)
    figure.canvas.draw()
    figure.canvas.flush_events()
    if(error > 40):
        steps = 1
    elif(error < 40 and error > 30):
        steps = 5
    elif(error < 30 and error > 20):
        steps = 10
    elif(error < 20 and error >10):
        TRAINING_FUNCT = infn_train2
        steps = 100
    elif (error < 10 and error > 8):
        TRAINING_FUNCT = infn_train3
        steps = 500
    elif (error < 8 and error > 6):
        TRAINING_FUNCT = infn_lateStage
        steps = 1000
    elif (error < 6):
        steps = 2000







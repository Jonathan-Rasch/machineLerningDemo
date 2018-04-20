import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dataGen
import logging
import math
logging.getLogger().setLevel(logging.INFO)
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
df_train,df_test,x_scalar,y_scalar = dataGen.getData(10000,0.3)
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
infn_train = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['delivery_time_min']],num_epochs=500,batch_size=128,shuffle=True)
infn_test = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],y=df_test[['delivery_time_min']],num_epochs=1,shuffle=False)
infn_pred = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],num_epochs=1,shuffle=False)
########################################################################################################################
# constructing model
########################################################################################################################
model = tf.estimator.DNNRegressor(hidden_units=[16,16,8],feature_columns=f_cols,dropout=0.1,activation_fn=tf.nn.elu)
########################################################################################################################
# training
########################################################################################################################
model.train(input_fn=infn_train, steps=1)
error = evaluate(model)
errors = []
index = 0
steps = 1
while(error>5.5):
    model.train(input_fn=infn_train,steps=steps)
    index += 1
    error = evaluate(model)
    errors.append(error)
    if(error > 20):
        steps = 1
    elif(error < 20 and error >10):
        steps = 100
    elif (error < 10 and error > 7):
        steps = 500
    elif (error < 7):
        steps = 1000
plt.plot(errors)
plt.show()





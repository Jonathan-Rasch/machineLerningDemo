import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fashion import dataGen
from fashion.dataGen import normalize
import random
from fashion.DNN_visual import NeuralNetwork
import time
########################################################################################################################
# PARAMETERS
########################################################################################################################
#logging.getLogger().setLevel(logging.INFO)
RANDOM_STATE = 101
NOISE_LEVEL = 0.2
np.random.seed(RANDOM_STATE)
tf.set_random_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
columns=['day_of_the_week','time_of_day','weather_type','temperature','colorful_percentage']
########################################################################################################################
# Supporting functions
########################################################################################################################

def evaluate(model):
    prediction_lst = model.predict(input_fn=infn_pred)
    print(model.evaluate(input_fn=infn_test))
    predictions = []
    for pred in prediction_lst:
        value_raw = pred['predictions']
        value = y_scalar.inverse_transform(value_raw.reshape(1,-1))
        predictions.append(value[0][0])
    #computing average error
    error_sum = 0
    for index,actual_value in enumerate(y_scalar.inverse_transform(df_test[['colorful_percentage']].values)):
        predicted = predictions[index]
        if (predicted < actual_value):
            error = actual_value - predicted
        else:
            error = predicted - actual_value
        error_sum += error
    avg_err = error_sum/len(predictions)
    print(" AVERAGE ERROR: " + str(avg_err))
    return avg_err

def pltErrGraph(x_axis,errors):
    fig = plt.figure(2)
    fig.clear()
    axis_error = error_figure.add_subplot(111)
    axis_error.set_autoscaley_on(True)
    axis_error.set_title("Average model error.")
    plt.xlabel("Number of training iterations")
    plt.ylabel("Percentage error in prediction")
    axis_error.plot(x_axis, errors)
    # plt.annotate('MODE: Training', (0, 0), (25, 200), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.annotate('AVG ERROR: {}% '.format(round(error[0] * 100,2)), (0, 0), (25, 210), xycoords='axes fraction',
                 textcoords='offset points', va='top')
    plt.annotate('ITERATION: {} '.format(index), (0, 0), (25, 220), xycoords='axes fraction',
                 textcoords='offset points', va='top')
    # plt.annotate('PREDICTION: {}% '.format(" -- "),
    #              (0, 0), (25, 230), xycoords='axes fraction', textcoords='offset points', va='top')
    # plt.annotate('INPUTS: temp={}C,type={},day={},time={}'.format(" -- "," -- ", " -- "," -- "),
    #              (0, 0), (25, 240), xycoords='axes fraction', textcoords='offset points', va='top')
    error_figure.canvas.draw()
    error_figure.canvas.flush_events()

def scale2dArr(array2d, arr_min, arr_max, scaleMin = 0, scaleMax = 1):
    diff = arr_max - arr_min
    scaledArr = []
    for array in array2d:
        sub_scaledArr = []
        for val in array:
            value = abs(val)
            sub_scaledArr.append(scaleMin + (value/diff)*scaleMax)
        scaledArr.append(sub_scaledArr)
    return scaledArr
########################################################################################################################
# getting data
########################################################################################################################
print("OBTAINING DATA...")
df_train,df_test,x_scalar,y_scalar = dataGen.getData(2500, 0.3)
########################################################################################################################
# generating graph
########################################################################################################################
error_figure = plt.figure(2)
axis_error = error_figure.add_subplot(111)
axis_error.set_autoscaley_on(True)
axis_error.set_title("Average model error.")
plt.xlabel("Number of training iterations")
plt.ylabel("Error in prediction")
plt.show(block=False)
########################################################################################################################
# constructing feature columns
########################################################################################################################
feature_columns = ['time_of_day','day_of_the_week','weather_type','temperature']
f_time_of_day = tf.feature_column.numeric_column(key='time_of_day')
f_temperature = tf.feature_column.numeric_column(key='temperature')
f_weather_type = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(key='day_of_the_week', vocabulary_list=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), dimension=3)
f_day = tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(key='weather_type', vocabulary_list=['sunny','overcast','rain/snow','storm']), dimension=2)
f_cols = [f_time_of_day, f_temperature, f_weather_type, f_day]
########################################################################################################################
# constructing input functions
########################################################################################################################
infn_train1 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=1,shuffle=True)
infn_train2 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=2,shuffle=True)
infn_train3 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=4,shuffle=True)
infn_train4 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=8,shuffle=True)
infn_train5 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=16,shuffle=True)
infn_train6 = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=32,shuffle=True)
infn_lateStage = tf.estimator.inputs.pandas_input_fn(x=df_train[feature_columns],y=df_train[['colorful_percentage']],num_epochs=500,batch_size=64,shuffle=True)
TRAINING_FUNCT = infn_train1
infn_test = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],y=df_test[['colorful_percentage']],num_epochs=1,shuffle=False)
infn_pred = tf.estimator.inputs.pandas_input_fn(x=df_test[feature_columns],num_epochs=1,shuffle=False)
########################################################################################################################
# constructing model
########################################################################################################################
steps = 1
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 500,
    keep_checkpoint_max = 1
)
model = tf.estimator.DNNRegressor(hidden_units=[16,16,8],feature_columns=f_cols,dropout=0.1,activation_fn=tf.nn.elu,config=my_checkpointing_config)
########################################################################################################################
# training
########################################################################################################################
model.train(input_fn=TRAINING_FUNCT, steps=1)
nn = NeuralNetwork([16,16,8],7,1) # 5 because vehicle type feature column has 2 dimensions
nn.updateLayerWeights(4, [[1],[1],[1],[1],[1],[1],[1],[1]]) # for output layer

# obtaining weight vectors
weights_hidden0_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_0/kernel")))
weights_hidden1_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_1/kernel")))
weights_hidden2_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_2/kernel")))
combined_arrays = np.concatenate((weights_hidden0_unscaled.flatten(),weights_hidden1_unscaled.flatten(),weights_hidden2_unscaled.flatten())).reshape(-1,1)
arr_min = None
arr_max = None
for val in combined_arrays:
    value = val[0]
    if(arr_min == None or arr_min > value):
        arr_min = value
    if(arr_max == None or arr_max < value):
        arr_max = value
weights_hidden0 = scale2dArr(weights_hidden0_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
weights_hidden1 = scale2dArr(weights_hidden1_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
weights_hidden2 = scale2dArr(weights_hidden2_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
nn.updateLayerWeights(1, weights_hidden0) # hidden layer 0 is the 1 st layer of network (layer 0 is input layer)
nn.updateLayerWeights(2, weights_hidden1)
nn.updateLayerWeights(3, weights_hidden2)
nn.draw()
plt.pause(60)
#input("Press any key to begin training.")
error = evaluate(model)
errors = []
x_axis = []
index = 0
input()
while(error>=0.05):
    # training model
    model.train(input_fn=TRAINING_FUNCT,steps=steps)
    # obtaining weight vectors
    weights_hidden0_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_0/kernel")))
    weights_hidden1_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_1/kernel")))
    weights_hidden2_unscaled = np.array(list(model.get_variable_value("dnn/hiddenlayer_2/kernel")))
    combined_arrays = np.concatenate((weights_hidden0_unscaled.flatten(),weights_hidden1_unscaled.flatten(),weights_hidden2_unscaled.flatten())).reshape(-1,1)
    arr_min = None
    arr_max = None
    for val in combined_arrays:
        value = val[0]
        if(arr_min == None or arr_min > value):
            arr_min = value
        if(arr_max == None or arr_max < value):
            arr_max = value
    weights_hidden0 = scale2dArr(weights_hidden0_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
    weights_hidden1 = scale2dArr(weights_hidden1_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
    weights_hidden2 = scale2dArr(weights_hidden2_unscaled, arr_min=arr_min, arr_max=arr_max,scaleMin=0.1,scaleMax=2)
    nn.updateLayerWeights(1, weights_hidden0) # hidden layer 0 is the 1 st layer of network (layer 0 is input layer)
    nn.updateLayerWeights(2, weights_hidden1)
    nn.updateLayerWeights(3, weights_hidden2)
    nn.draw() #nn.updateFigure() # updating nn graph
    # computing average error
    error = evaluate(model)
    # making changes to graphs
    index += steps
    x_axis.append(index)
    errors.append(error)
    pltErrGraph(x_axis,errors)
    # adjusting info screen
    #addjusting step number
    if(error > 0.25):
        steps = 5
        TRAINING_FUNCT=infn_train1
    elif(error < 0.25 and error >= 0.15):
        steps = 10
        TRAINING_FUNCT = infn_train3
    elif(error < 0.15 and error >= 0.10):
        steps = 20
        TRAINING_FUNCT = infn_train4
    elif(error < 0.10 and error >= 0.08):
        steps = 40
        TRAINING_FUNCT = infn_train5
    elif (error < 0.08 and error > 0.06):
        TRAINING_FUNCT = infn_train6
        steps = 80
    elif (error < 0.06):
        TRAINING_FUNCT = infn_lateStage
        steps += 10
# print("\nPREDICTION MODE\n")
# # while(True):
# #     isexit = input("type x to abort, or any other key to continue: ") == "x"
# #     if (isexit):
# #         exit(0)
# #     # selecting drivers vehicle
# #     weather_types = ['sunny','overcast','rain/snow','storm']
# #     while(True):
# #         print("Select weather type: (0: sunny, 1: overcast, 2: rain/snow 3: storm)")
# #         weather_type = input("weather: ")
# #         if(str(weather_type) in ['0','1','2','3']):
# #             weather_type = weather_types[int(weather_type)]
# #             break
# #         else:
# #             print("invalid input.")
# #     # selecting distance to customer
# #     while(True):
# #         print("Select temperature: (number between 0 and 35 (inclusive) C)")
# #         temperature = input("temperature: ")
# #         try:
# #             temperature = float(temperature)
# #             if (temperature >= 0 and temperature <= 35):
# #                 break
# #             else:
# #                 print("invalid input.")
# #         except:
# #             print("invalid input.")
# #     # selecting time of day
# #     while (True):
# #         print("Select hour of day: (number between 0 and 24 (exclusive) )")
# #         time = input("time: ")
# #         try:
# #             time = float(time)
# #             if (time >= 0 and time < 24):
# #                 break
# #             else:
# #                 print("invalid input.")
# #         except:
# #             print("invalid input.")
# #     # selecting order size
# #     while (True):
# #         print("Select day: (Mon,Tue,Wed,Thu,Fri,Sat,Sun) )")
# #         day = input("day: ")
# #         try:
# #             if (day.lower().title() in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']):
# #                 break
# #             else:
# #                 print("invalid input.")
# #         except:
# #             print("invalid input.")
#     # creating data frame and input function
# inputs = []
# inputs.append({'time_of_day':12,'day_of_the_week':'Sun','weather_type':'sunny','temperature':25})
# for input in inputs:
#     row_dict = input
#     df = pd.DataFrame(data=None,columns=['day_of_the_week','time_of_day','weather_type','temperature'])
#     df = df.append(other=row_dict, ignore_index=True)
#     df_scaled = normalize(df,columns=['time_of_day','temperature'],col_scalar=x_scalar)
#     #data = pd.concat([data_unscaled, data_scaled], axis=1)
#     pred_in_fn = tf.estimator.inputs.pandas_input_fn(x=df_scaled[0],batch_size=1,num_epochs=1,shuffle=False)
#     prediction_lst = list(model.predict(input_fn=pred_in_fn))
#     for pred in prediction_lst:
#         value_raw = pred['predictions']
#         value = y_scalar.inverse_transform(value_raw.reshape(1,-1))
#     # plotting prediction
#     fig = plt.figure(2)
#     fig.clear()
#     axis_error = error_figure.add_subplot(111)
#     axis_error.set_autoscaley_on(True)
#     axis_error.set_title("Average model error.")
#     plt.xlabel("Number of training iterations")
#     plt.ylabel("Percentage error in prediction")
#     axis_error.plot(x_axis, errors)
#     plt.annotate('MODE: Prediction', (0, 0), (25, 200), xycoords='axes fraction', textcoords='offset points', va='top')
#     plt.annotate('AVG ERROR: {}% '.format(round(error[0] * 100)), (0, 0), (25, 210), xycoords='axes fraction',
#                  textcoords='offset points', va='top')
#     plt.annotate('ITERATION: {} '.format(index), (0, 0), (25, 220), xycoords='axes fraction',
#                  textcoords='offset points', va='top')
#     plt.annotate('PREDICTION: {}% '.format(round(value[0][0]*100)),
#                  (0, 0), (25, 230), xycoords='axes fraction', textcoords='offset points', va='top')
#     plt.annotate('INPUTS: temp={}C,type={},day={},time={}'.format(input['temperature'], input['weather_type'],
#                                            input['day_of_the_week'], input['time_of_day']),
#                  (0, 0), (25, 240), xycoords='axes fraction', textcoords='offset points', va='top')
#     error_figure.canvas.draw()
#     error_figure.canvas.flush_events()
#     print("-------------------------------------------------------------")
#     print("Input Parameters: time={}, day={}, temperature={}, weather_type={}".format(input['time_of_day'],input['day_of_the_week'],input['temperature'],input['weather_type']))
#     print("Predicted percentage of colorful clothes: {} %.".format(round(value[0][0]*100)))
#     print("-------------------------------------------------------------")








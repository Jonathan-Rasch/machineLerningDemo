import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pizza import dataGen
from pizza.dataGen import normalize
import random
from pizza.DNN_visual import NeuralNetwork
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
    for index,actual_value in enumerate(y_scalar.inverse_transform(df_test[['delivery_time_min']].values)):
        predicted = predictions[index]
        if (predicted < actual_value):
            error = actual_value - predicted
        else:
            error = predicted - actual_value
        error_sum += error
    avg_err = error_sum/len(predictions)
    print(" AVERAGE ERROR: " + str(avg_err) + " minutes")
    return avg_err

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
df_train,df_test,x_scalar,y_scalar = dataGen.getData(1000, 0.3, visualise=False, noiseLevel=NOISE_LEVEL)
########################################################################################################################
# generating graph
########################################################################################################################
error_figure = plt.figure(2)
axis_error = error_figure.add_subplot(111)
axis_error.set_autoscaley_on(True)
#axis_error.set_ylim([0,60])
axis_error.set_yticks(list(range(0,500,5)),minor=False)
axis_error.set_title("Avg model error (in minutes) vs training iterations ")
plt.xlabel("number of training iterations")
plt.ylabel("error (minutes) in predicted delivery time vs actual delivery time")
plt.figure(1)
# for axis in scatterMatrix.flatten():
#     axis.plot()
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
    keep_checkpoint_max = 2       # Retain the 10 most recent checkpoints.
    #model_dir='pizza'
)
model = tf.estimator.DNNRegressor(hidden_units=[16,16,8],feature_columns=f_cols,dropout=0.1,activation_fn=tf.nn.elu,config=my_checkpointing_config)
########################################################################################################################
# training
########################################################################################################################
model.train(input_fn=TRAINING_FUNCT, steps=1)
nn = NeuralNetwork([16,16,8],5,1) # 5 because vehicle type feature column has 2 dimensions
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
plt.pause(10)
input("Press any key to begin training.")
error = evaluate(model)
errors = []
x_axis = []
index = 0
while(error>5):
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
    if(error>60):
        continue
    # making changes to graphs
    plt.figure(2)
    index += steps
    x_axis.append(index)
    errors.append(error)
    line, = axis_error.plot(x_axis,errors)
    error_figure.canvas.draw()
    error_figure.canvas.flush_events()
    # addjusting step number
    if(error > 40):
        steps = 5
    elif(error < 40 and error > 30):
        steps = 10
    elif(error < 30 and error > 20):
        steps = 50
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
print("\nPREDICTION MODE\n")
while(True):
    isexit = input("type x to abort, or any other key to continue: ") == "x"
    if (isexit):
        exit(0)
    # selecting drivers vehicle
    vehicle_types = ['none', 'car', 'scooter', 'bicycle']
    while(True):
        print("Select drivers vehicle type: (0: None, 1: car, 2: scooter 3: bicycle)")
        vehicle = input("vehicle: ")
        if(str(vehicle) in ['0','1','2','3']):
            vehicle = vehicle_types[int(vehicle)]
            break
        else:
            print("invalid input.")
    # selecting distance to customer
    while(True):
        print("Select distance to customer: (number between 0 and 20 (inclusive) Kilometers)")
        distance = input("distance: ")
        try:
            f_dist = float(distance)
            if (f_dist > 0 and f_dist <= 20):
                distance = float(distance)
                break
            else:
                print("invalid input.")
        except:
            print("invalid input.")
    # selecting time of day
    while (True):
        print("Select hour of day: (number between 0 and 24 (exclusive) )")
        time = input("time: ")
        try:
            f_time = float(time)
            if (f_time > 0 and f_time < 24):
                time = float(f_time)
                break
            else:
                print("invalid input.")
        except:
            print("invalid input.")
    # selecting order size
    while (True):
        print("Select order size: (number between 0 and 1 (inclusive) )")
        size = input("size: ")
        try:
            f_size = float(size)
            if (f_size > 0 and f_size < 24):
                size = float(f_size)
                break
            else:
                print("invalid input.")
        except:
            print("invalid input.")
    # creating data frame and input function
    columns = ['distance_km', 'order_size', 'vehicle_type', 'hour_of_day']
    row_dict = {'distance_km':distance,'order_size':size,'vehicle_type':vehicle,'hour_of_day':time}
    df = pd.DataFrame(data=None,columns=columns)
    df = df.append(other=row_dict, ignore_index=True)
    df_scaled = normalize(df,columns=['distance_km','order_size','hour_of_day'],col_scalar=x_scalar)
    #data = pd.concat([data_unscaled, data_scaled], axis=1)
    pred_in_fn = tf.estimator.inputs.pandas_input_fn(x=df_scaled[0],batch_size=1,num_epochs=1,shuffle=False)
    prediction_lst = list(model.predict(input_fn=pred_in_fn))
    for pred in prediction_lst:
        value_raw = pred['predictions']
        value = y_scalar.inverse_transform(value_raw.reshape(1,-1))
    print("PREDICTED DELIVERY TIME: {} minutes.".format(value[0][0]))
    print("-------------------------------------------------------------")





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
import math

def generatePizzaData(numDataPoints = 10000,noiseLevel=1) -> pd.DataFrame:
    np.random.seed(101)
    df = pd.DataFrame(data=None,columns=['distance_km','order_size','vehicle_type','hour_of_day','delivery_time_min'])
    for point_index in range(0,numDataPoints):
        row_dict = {}
        ########################################################################################################################
        # Ordersize (larger order means more time to prepare)
        ########################################################################################################################
        orderSize = np.random.uniform()
        row_dict['order_size'] = orderSize
        preperationTime = math.fabs(orderSize * 40 + 2.5 * np.random.normal() * noiseLevel)  # maximum, +5 is min preperation time and quality check
        ########################################################################################################################
        # delivery distance
        ########################################################################################################################
        MAX_DELIVERY_DISTANCE_KM = 20
        distanceToCustomer_km = np.random.uniform() * MAX_DELIVERY_DISTANCE_KM # max distance km
        row_dict['distance_km'] = distanceToCustomer_km
        timeDueToDistance = 0.1145*(distanceToCustomer_km**2)-0.915*(distanceToCustomer_km)+5*np.random.normal()*noiseLevel
        ########################################################################################################################
        # delivery time due to time of day (traffic and demand could cause this)
        ########################################################################################################################
        hour_of_day = np.random.uniform(low=0,high=1)
        row_dict['hour_of_day'] = hour_of_day*24
        timeDueToHour = 20*hour_of_day #added time due to traffic etc
        ########################################################################################################################
        # driver vehicle (how well the driver can get around)
        ########################################################################################################################
        vehicle_types = ['none','car','scooter','bicycle']
        vehicle = np.random.randint(low=0,high=4)
        row_dict['vehicle_type'] = vehicle_types[vehicle]
        if(vehicle == 0): #walking
            timeDueToVehicle = (distanceToCustomer_km / (5.0*np.random.uniform(low=0.8,high=1))) * 60 #assuming avg walking speed of 5Kmh
        elif(vehicle == 1): # car
            timeDueToVehicle = (distanceToCustomer_km / (10.0 * np.random.uniform(low=0.8, high=1))) * 60
        elif(vehicle == 2): # scooter
            timeDueToVehicle = (distanceToCustomer_km / (20.00*np.random.uniform(low=0.8,high=1))) * 60
        elif (vehicle == 3): # bicicle
            timeDueToVehicle = (distanceToCustomer_km / (30.0 * np.random.uniform(low=0.8, high=1))) * 60
        ########################################################################################################################
        # Summing times
        ########################################################################################################################
        # adding to dataframe
        row_dict['delivery_time_min'] = timeDueToDistance + preperationTime + timeDueToHour + timeDueToVehicle
        df = df.append(other=row_dict, ignore_index=True)
    return df

def normalize(data: pd.DataFrame,columns=None) -> pd.DataFrame:
    unused_cols = []
    if(columns == None):
        columns = data.columns
    for col in data.columns:
        if(not col in columns ):
            unused_cols.append(col)
    scalar = MinMaxScaler().fit(data[columns])
    data_ndarr = scalar.transform(data[columns])
    data_scaled = pd.DataFrame(data=data_ndarr,columns=columns)
    data_unscaled = data[unused_cols].reset_index(drop=True)
    data = pd.concat([data_unscaled,data_scaled],axis=1)
    return (data,scalar)

def transform(data: pd.DataFrame,scalar,columns = None):
    unused_cols = []
    if (columns == None):
        columns = data.columns
    for col in data.columns:
        if (not col in columns):
            unused_cols.append(col)
    scaled = pd.DataFrame(data=scalar.transform(data[columns]),columns=columns)
    data = pd.concat([data[unused_cols].reset_index(drop=True),scaled],axis=1)
    return data

def getData(n=10000,test_percentage=0.3,visualise=False):
    data = generatePizzaData(n, 0.2)
    features = data.drop(labels=['delivery_time_min'],axis=1)
    labels = data[['delivery_time_min']]
    X_train_raw, X_test_raw , y_train_raw, y_test_raw = train_test_split(features,labels,test_size=test_percentage,shuffle=False,random_state=101)
    x_train, x_scalar = normalize(X_train_raw,['distance_km','order_size','hour_of_day'])
    y_train, y_scalar = normalize(y_train_raw,['delivery_time_min'])
    x_test = transform(X_test_raw,x_scalar,['distance_km','order_size','hour_of_day'])
    y_test = transform(y_test_raw,y_scalar,['delivery_time_min'])
    df_x_train = pd.DataFrame(data=x_train,columns=features.columns)
    df_y_train = pd.DataFrame(data=y_train,columns=labels.columns)
    df_x_test = pd.DataFrame(data=x_test,columns=features.columns)
    df_y_test = pd.DataFrame(data=y_test,columns=labels.columns)
    df_train = pd.concat([df_x_train,df_y_train],axis=1)
    df_test = pd.concat([df_x_test, df_y_test], axis=1)
    if(visualise):
        data.replace(to_replace='none', value=0, inplace=True)
        data.replace(to_replace='car', value=1, inplace=True)
        data.replace(to_replace='scooter', value=2, inplace=True)
        data.replace(to_replace='bicicle', value=3, inplace=True)
        # data.plot(x='distance_km', y='delivery_time_min', style='x')
        # data.plot(x='order_size', y='delivery_time_min', style='x')
        # data.plot(x='vehicle_type', y='delivery_time_min', style='x')
        # data.plot(x='hour_of_day', y='delivery_time_min', style='x')
        # scatter matrix
        axis_scatter = pd.scatter_matrix(data,alpha=0.2,diagonal='kde')
        return (df_train, df_test, x_scalar, y_scalar,axis_scatter)
    else:
        return (df_train,df_test,x_scalar,y_scalar)

if __name__ == "__main__":
    data = generatePizzaData(1000,1)
    data.replace(to_replace='none', value=0, inplace=True)
    data.replace(to_replace='car', value=1, inplace=True)
    data.replace(to_replace='scooter', value=2, inplace=True)
    data.replace(to_replace='bicycle', value=3, inplace=True)
    plt.figure(1)
    # data.plot(x='distance_km',y='delivery_time_min',style='x')
    # data.plot(x='order_size', y='delivery_time_min', style='x')
    # data.plot(x='vehicle_type', y='delivery_time_min', style='x')
    # data.plot(x='hour_of_day', y='delivery_time_min', style='x')
    #scatter matrix
    axis_scatter = pd.scatter_matrix(data,diagonal='kde')
    # normalized_data,scalar = normalize(data)
    # corr_values = normalized_data.corr().values
    # setting background color of axes bg
    # cmap = 'hot'
    # for index,sub_axis in enumerate(axis_scatter.flatten()):
    #     color = plt.cm.hot(corr_values.flatten()[index])
    #     sub_axis.set_facecolor(color)
    plt.show()
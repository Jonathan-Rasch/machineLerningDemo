import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
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
        preperationTime = math.fabs(
        orderSize * 30 + 10 * np.random.normal() * noiseLevel)  # maximum, +5 is min preperation time and quality check
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
        vehicle_types = ['none','car','scooter','bicicle']
        vehicle = np.random.randint(low=0,high=4)
        row_dict['vehicle_type'] = vehicle
        if(vehicle == 0): #walking
            timeDueToVehicle = (distanceToCustomer_km / 5.0*np.random.uniform(low=0.3,high=1)) * 60 #assuming avg walking speed of 5Kmh
        elif(vehicle == 1): # car
            timeDueToVehicle = (distanceToCustomer_km / 10.0 * np.random.uniform(low=0.3, high=1)) * 60
        elif(vehicle == 2): # scooter
            timeDueToVehicle = (distanceToCustomer_km / 20.00*np.random.uniform(low=0.3,high=1)) * 60
        elif (vehicle == 3): # bicicle
            timeDueToVehicle = (distanceToCustomer_km / 30.0 * np.random.uniform(low=0.3, high=1)) * 60
        ########################################################################################################################
        # Summing times
        ########################################################################################################################
        # adding to dataframe
        row_dict['delivery_time_min'] = timeDueToDistance + preperationTime + timeDueToHour + timeDueToVehicle
        df = df.append(other=row_dict, ignore_index=True)
    return df

def normalize(data: pd.DataFrame) -> pd.DataFrame:
    scalar = MinMaxScaler().fit(data)
    data_ndarr = scalar.transform(data)
    data = pd.DataFrame(data=data_ndarr,columns=data.columns)
    return (data,scalar)

def getData(n=10000):
    data = generatePizzaData(n, 1)
    return normalize(data)

if __name__ == "__main__":
    data = generatePizzaData(1000,2)
    data.plot(x='distance_km',y='delivery_time_min',style='x')
    data.plot(x='order_size', y='delivery_time_min', style='x')
    data.plot(x='vehicle_type', y='delivery_time_min', style='x')
    data.plot(x='hour_of_day', y='delivery_time_min', style='x')
    #scatter matrix
    axis_scatter = pd.scatter_matrix(data)
    normalized_data,scalar = normalize(data)
    corr_values = normalized_data.corr().values
    # setting background color of axes bg
    # cmap = 'hot'
    # for index,sub_axis in enumerate(axis_scatter.flatten()):
    #     color = plt.cm.hot(corr_values.flatten()[index])
    #     sub_axis.set_facecolor(color)
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def generatePizzaData(numDataPoints = 10000,noiseLevel=1) -> pd.DataFrame:
    np.random.seed(101)
    df = pd.DataFrame(data=None,columns=['distance_km','order_size','delivery_time_min'])
    for point_index in range(0,numDataPoints):
        row_dict = {}
        orderSize = np.random.uniform()
        row_dict['order_size'] = orderSize
        preperationTime = math.fabs(orderSize * 30 + 10*np.random.normal()*noiseLevel) #maximum, +5 is min preperation time and quality check
        MAX_DELIVERY_DISTANCE = 20
        distanceToCustomer_km = np.random.uniform() * MAX_DELIVERY_DISTANCE # max distance km
        row_dict['distance_km'] = distanceToCustomer_km
        timeDueToDistance = 0.1145*(distanceToCustomer_km**2)-0.915*(distanceToCustomer_km)+5*np.random.normal()*noiseLevel
        hour_of_day = np.random.uniform(low=0,high=24)
        #timeDueToHour =
        # adding to dataframe
        row_dict['delivery_time_min'] = timeDueToDistance + preperationTime
        df = df.append(other=row_dict, ignore_index=True)
    return df

if __name__ == "__main__":
    data = generatePizzaData(1000,2)
    pd.scatter_matrix(data)
    data.plot(x='distance_km',y='delivery_time_min',style='x')
    plt.show()
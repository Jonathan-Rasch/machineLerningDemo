import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib
import math

########################################################################################################################
# DATAGEN PARAMETERS
########################################################################################################################
MAXIMUM_TEMPERATURE_C = 35
# note: below parameters should add to 1
TEMPERATURE_WEIGHT = 0.30
WEATHER_TYPE_WEIGHT = 0.45
DAY_OF_WEEK_WEIGHT = 0.05
TIME_OF_DAY_WEIGHT = 0.15
RANDOM_ERROR_WEIGHT = 0.05

def generateData(numDataPoints = 10000) -> pd.DataFrame:
    np.random.seed(101)
    df = pd.DataFrame(data=None,columns=['day_of_the_week','time_of_day','weather_type','temperature','colorful_percentage'])
    for point_index in range(0,numDataPoints):
        row_dict = {}
        ########################################################################################################################
        # Time of day
        ########################################################################################################################
        time_of_day = np.random.random()
        time = time_of_day * 24
        time_contribution = (math.sin(math.pi * 4 * time_of_day + (math.pi * 0.5 * np.random.uniform(low=-1, high=1))) + 1) / 2
        #time_contribution += (RANDOM_ERROR_WEIGHT)*np.random.uniform(low=-1,high=1)
        row_dict['time_of_day'] = time
        ########################################################################################################################
        # temperature impact
        ########################################################################################################################
        temp_before_considering_time = np.random.random()
        temperature_norma = temp_before_considering_time*0.6 + 0.4*temp_before_considering_time*(2*time_of_day if time_of_day < 0.5 else (1-2*(time_of_day-0.5))) + RANDOM_ERROR_WEIGHT*np.random.uniform(low=-1, high=1)
        temp_contribution = 1.5 * (temperature_norma**2) - 0.5 * temperature_norma - 0.2# random quadratic to model the effect of temp
        #temp_contribution += (RANDOM_ERROR_WEIGHT) * np.random.uniform(low=-1, high=1)
        temperature_C = temperature_norma * MAXIMUM_TEMPERATURE_C # value for the dataframe
        row_dict['temperature'] = temperature_C
        ########################################################################################################################
        # weather type impact
        ########################################################################################################################
        sun_chance = ((30/19)*temperature_norma - (11/19))  if (temperature_norma >= 0.43) else 0.1 # chance that its sunny
        weather_types = {'sunny':1,'overcast':0.5,'rain/snow':0.5,'storm':0}
        if(sun_chance >= np.random.random()):
            weather_type = 'sunny'
        else:
            overcast_chance = 0.6
            if(overcast_chance >= np.random.random()):
                weather_type = 'overcast'
            else:
                storm_chance = 0.7*temperature_norma
                if(storm_chance >= np.random.random()):
                    weather_type = 'storm'
                else:
                    weather_type = 'rain/snow'
        weather_type_contribution = weather_types[weather_type]
        #weather_type_contribution += (RANDOM_ERROR_WEIGHT) * np.random.uniform(low=-1, high=1)
        row_dict['weather_type'] = weather_type
        ########################################################################################################################
        # day of the week impact
        ########################################################################################################################
        day_of_the_week = {'Mon':0.1,'Tue':0.25,'Wed':0.35,'Thu':0.25,'Fri':0.5,'Sat':0.75,'Sun':1.0}
        day = list(day_of_the_week.keys())[np.random.randint(low=0,high=7)]
        day_contribution = day_of_the_week[day]
        #day_contribution += (RANDOM_ERROR_WEIGHT) * np.random.uniform(low=-1, high=1)
        row_dict['day_of_the_week'] = day
        ########################################################################################################################
        # FINALISING
        ########################################################################################################################
        # adding to dataframe
        row_dict['colorful_percentage'] = temp_contribution*TEMPERATURE_WEIGHT + weather_type_contribution*WEATHER_TYPE_WEIGHT + day_contribution*DAY_OF_WEEK_WEIGHT + time_contribution*TIME_OF_DAY_WEIGHT + np.random.random()*RANDOM_ERROR_WEIGHT
        df = df.append(other=row_dict, ignore_index=True)
    return df

def normalize(data: pd.DataFrame,columns=None,col_scalar = None) -> pd.DataFrame:
    unused_cols = []
    if(columns == None):
        columns = data.columns
    for col in data.columns:
        if(not col in columns ):
            unused_cols.append(col)
    if(col_scalar == None):
        scalar = MinMaxScaler().fit(data[columns])
    else:
        scalar = col_scalar
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

def getData(n=10000,test_percentage=0.3):
    data = generateData(n)
    features = data.drop(labels=['colorful_percentage'],axis=1)
    labels = data[['colorful_percentage']]
    X_train_raw, X_test_raw , y_train_raw, y_test_raw = train_test_split(features,labels,test_size=test_percentage,shuffle=False,random_state=101)
    x_train, x_scalar = normalize(X_train_raw,['time_of_day','temperature'])
    y_train, y_scalar = normalize(y_train_raw,['colorful_percentage'])
    x_test = transform(X_test_raw,x_scalar,['time_of_day','temperature'])
    y_test = transform(y_test_raw,y_scalar,['colorful_percentage'])
    df_x_train = pd.DataFrame(data=x_train,columns=features.columns)
    df_y_train = pd.DataFrame(data=y_train,columns=labels.columns)
    df_x_test = pd.DataFrame(data=x_test,columns=features.columns)
    df_y_test = pd.DataFrame(data=y_test,columns=labels.columns)
    df_train = pd.concat([df_x_train,df_y_train],axis=1)
    df_test = pd.concat([df_x_test, df_y_test], axis=1)
    return (df_train,df_test,x_scalar,y_scalar)

if __name__ == "__main__":
    data = generateData(2000)
    # replacing day of the week values
    data.replace(to_replace='Mon', value=0, inplace=True)
    data.replace(to_replace='Tue', value=1, inplace=True)
    data.replace(to_replace='Wed', value=2, inplace=True)
    data.replace(to_replace='Thu', value=3, inplace=True)
    data.replace(to_replace='Fri', value=4, inplace=True)
    data.replace(to_replace='Sat', value=5, inplace=True)
    data.replace(to_replace='Sun', value=6, inplace=True)
    # replacing weather type values
    data.replace(to_replace='sunny', value=0, inplace=True)
    data.replace(to_replace='overcast', value=1, inplace=True)
    data.replace(to_replace='rain/snow', value=2, inplace=True)
    data.replace(to_replace='storm', value=3, inplace=True)
    #scatter matrix
    axis_scatter = pd.scatter_matrix(data,diagonal='kde')
    plt.show()
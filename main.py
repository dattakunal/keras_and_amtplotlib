import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.indexes.datetimes import date_range
from pandas.tseries import frequencies
import seaborn as sns
import datetime as dt
from pandas.tseries.offsets import BDay

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

def read_and_create_dataset(call_types, consider_weekend=1):
    file_name = 'SBOSS_35956_CDR_Counts_By_Date.xlsx'
    column_names = ['Call_Date', 
                    'total_call_Count', 
                    'inbound_int_count',
                    'inbound_ext_count',
                    'outbound_int_count',
                    'outbound_ext_count',
                    'inbound_chg_count',
                    'outbound_chg_count',
                    ]

    raw_dataset = pd.read_excel(file_name, 
                                names=column_names,
                                na_values='?',
                                comment='\t',
                                header=2)

    train_dataset = raw_dataset.copy()
    train_dataset = train_dataset.dropna()

    if consider_weekend==2:
        isBusinessDay = BDay().onOffset
        match_series = pd.to_datetime(train_dataset['Call_Date']).map(isBusinessDay)
        train_dataset = train_dataset[match_series]
        #print(train_dataset.head(20))

    train_dataset[call_types[1]] = train_dataset['inbound_int_count'] + train_dataset['inbound_ext_count']
    train_dataset[call_types[2]] = train_dataset['outbound_int_count'] + train_dataset['outbound_ext_count']
    train_dataset[call_types[3]] = train_dataset['inbound_chg_count'] + train_dataset['outbound_chg_count']

    train_dataset['Call_Date']= pd.to_datetime(train_dataset['Call_Date'], errors='coerce')
    train_dataset['Call_Date']=train_dataset['Call_Date'].map(dt.datetime.toordinal)

    return train_dataset

def create_feature_and_model(train_dataset):
    train_features = train_dataset.copy()

    call_dates = np.array(train_features['Call_Date'])

    call_date_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    call_date_normalizer.adapt(call_dates)

    call_date_model = build_and_compile_model(call_date_normalizer)

    call_date_model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')

    return call_date_model

def train_and_predict_model(call_date_model, call_type):
    train_features = train_dataset.copy()
    train_labels = train_features.pop(call_type)

    history = call_date_model.fit(
        train_features['Call_Date'],
        train_labels,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split = 0.2)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    from_date = dt.date(year=2022, month=1, day=1)
    to_date = dt.date(year=2023, month=12, day=31)

    freq = 250

    x = tf.linspace(from_date.toordinal(), to_date.toordinal(), freq)
    xa = pd.date_range(from_date, to_date, periods=freq)
    
    y = call_date_model.predict(x)

    _, axis = plt.subplots(1, 2)

    ylabel = ' '.join(call_type.split('_')).title()
    
    # Trained feature
    train_features['Call_Date']=train_features['Call_Date'].map(dt.datetime.fromordinal)
    axis[0].ticklabel_format(style='plain')
    axis[0].scatter(train_features['Call_Date'], train_labels, marker='.')
    axis[0].set(xlabel='Call Date', ylabel=ylabel)
    axis[0].set_title('Training Dataset')

    # Prediction 
    axis[1].ticklabel_format(style='plain')
    axis[1].plot(xa, y)
    axis[1].set(xlabel='Call Date', ylabel=ylabel)
    axis[1].set_title('Predicted values')

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == "__main__":
    call_types = {1:'total_inbound_call', 
                2:'total_outbound_call', 
                3:'total_charged_call'}

    consider_weekend = int(input('Do you want to include weekends? 1 - Yes, 2 - No\n'))
    
    train_dataset = read_and_create_dataset(call_types, consider_weekend)
    
    model = create_feature_and_model(train_dataset)

    while True:
        print('Type the field you want to predict:')
        print('Inbound call: 1')
        print('Outbound call: 2')
        print('Charged call: 3')
        print('Exit: 0')

        flag = int(input())

        if(flag == 0):
            break

        call_type = call_types.get(flag)

        if(call_type == None):
            print('Invalid Input. Please try again')

        train_and_predict_model(model, call_type)






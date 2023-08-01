'''
Created on May 26, 2021

TemporalLstmDna applies LSTM to forecast time-series data. It provides the following
services:

    - Retrieve data from Hazelcast using query predicates
    - Split the retrieved data into train and test datasets
    - Fit a pandas Sequential model to the train dataset
    - Validate the model using the test dataset.
    - Save the model
    - Retrieve the model to forecast in real time
    - Update model info upon completion of generating each forecast
    - Generate a forecast for an individual observed data point
    - Generate forecasts for a batched list of observed data points

@author: dpark
'''

import json
import logging
import math
import os
import time

import numpy
from pandas import DataFrame
from pandas import Series
from pandas import concat
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
# For persisting scaler
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from statsmodels.tsa.stattools import adfuller

from keras.models import model_from_json
from keras.utils import disable_interactive_logging
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
import pandas as pd

from padogrid.bundle.dna.dna_client import Dna
from padogrid.bundle.util.class_util import get_class_name

logger = logging.getLogger('TemporalLstmDna')

class TemporalLstmDna(Dna):
     
    '''
    TemporalLstmDna performs linear regression on temporal data using 
    LSTM Recurrent Neural Network with the Keras library backed by 
    TensorFlow.
    
    There are two "run" methods that perform LSTM on data returned from
    executing queries:
        run_lstm()
        run_pql()
    
    Using TemporalLstmDna:
        
        1. Implement a subclass that inherits TemporaLstmDna and overrides
            the value_callback() method that returns the desired values.
            Also, override the run_lstm() method to invoke the super methods.
            This is required due to a limitation in Python.
        
            class SalesDna(TemporalLstmDna):
                def __init__(self):
                    if sys.version_info.major >= 3:
                        super().__init__()
                    else:
                        super(SalesDna, self).__init__()
                def value_callback(self, key, value):
                    factor = getattr(value, 'conversionFactor')
                    qty = tv['quantity']
                    value = factor * qty
                    return value
                def run_lstm(self, jparams, callback=None):
                    if sys.version_info.major >= 3:
                        return super().run_lstm(jparams, callback)
                    else:
                        return super(SalesDna, self).run_lstm(jparams, callback)
        
        2. Deploy SalesDna to the data nodes. For running locally, this step
            is not required.
        
        3. Login to Hazelcast
            client = hazelcast.HazelcastClient(cluster_name="jet", portable_factories=PortableFactoryImpl.factories(portable_factories))
            
        4. Run SalesDna
            Note that the model is automatically saved in the grid. You can use the 
            saved model to evaluate the test data by setting 'use_save_model=True' for local calls
            and useSaveModel for remote calls. Specify model_name/modelName to override
            the default model name.
            
            4.1 To fetch data from the grid and run LSTM locally:
                dna = TemporalLstmDna(client)
                country="Greece"
                jresult = dna.run_lstm_local(grid_path, 
                    where_clause,
                    time_attribute='shipDate',
                    value_attribute, 
                    use_saved_model=False, 
                    model_name='model_'+country, 
                    callback=value_callback)
                
            4.2 FUTURE: This option is currently unavailable (08/16/2021)
                To run LSTM remotely in the data nodes (Note that you can set 
                timeout to a small value and get the results later by checking
                the DNA run status. See step 5 for details):
                    jresponse = pado.invoke_dna(SalesDna.run_lstm, 
                        where_clause, 
                        value_attribute, 
                        useSavedModel=False, 
                        modelName='SalesDna', 
                        timeout=120000)
                    result = jresponse['result']
                    ml = result['ml']
                    jresult = None
                    for data_node in ml:
                        sid = data_node['sid']
                        if 'result' in data_node:
                            record = data_node['result']
                            if '__error' in record:
                                continue
                            else:
                                jresult = record
        5. Getting DNA run status (If the remote DNA run times out then you
            can asynchronously check the DNA run status for results as follows):
            response = pado.get_dna_status()
            if 'result' in response:
                jresult = response['result']
        6. View results. There are three lists with the same length.
            You can use pyplot to plot them.
            expected_list = jresult['Expected']
            predicted_list = jresult['Predicted']
            time_list = pd.to_datetime(jresult['Time'])
            pyplot.plot(time_list, expected_list)
            pyplot.plot(time_list, predicted_list)
            pyplot.show()
    '''
    df = None
    scaler = None
    train_scaled = None
    test_scaled = None
    model = None
    devider = 0
    index = None
    data_raw = None
    data_train = None
    data_test = None
    value_key = None
    
    # Time attribute name in the value objects queried from Hazelcast.
    time_attribute = None
    
    def __init__(self, feature=None, hazelcast_client=None, username=None, working_dir=None):
        '''
        Constructs a new TemporalLstmDna object.
        
        Args:
            working_dir: Working directory path. The model and log files are stored relative
                         to this directory.
        '''
        super().__init__(feature, hazelcast_client, username)
        self.working_dir = working_dir
    
    def __prep_data_frame(self, df, time_type=None):
        '''
        Prepares the specified data frame to be used to fit the model.

        1. Sorts the data frame by time
        2. Resets the data frame index
        3. Filters the data frame by invoking filter_data_frame()
        4. Sets the data frame index to the time column.
        5. Removes the time column from the data frame.
        6. Sets data_raw to df ['value']
        7. Sets the index to df.index
        
        Args:
            df: Data frame containing 'time' and 'value' columns of data.
        '''
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        df = df.reset_index(drop=True)
        df = self.filter_data_frame(df, time_type)
        print('filtered data len=%d' % len(df))
        df.index = df['time']
        del df['time']
        self.data_raw = df['value']
        self.index = df.index
        return df      
    
    def filter_data_frame(self, df, time_type='all'):
        '''
        Filters the specified data frame object.

        It creates a new filtered data frame object if time_type is not 'all'.

        This method can be overwritten to tailor the passed-in data frame. By default,
        it returns the passed-in data frame as is without making any changes.

        Args:
            df: DataFrame object with time sorted 'time' and 'value' lists.
            time_type:  'all' returns the same data frame, df, without any changes. Default is 'all'.
                        'date' returns accumulated date values.
                        'month' returns accumulated month values.
                        'year' returns accumulated year values.
                        Invalid type defaults to 'all'.
        '''
        if time_type != 'all' and time_type != 'hour' and time_type != 'date' and time_type != 'month' and time_type != 'year':
            time_type = 'all'
        if time_type == 'all':
            return df

        prev_date_time = None
        accumulator_value = 0.0
        df_time = df['time']
        df_value = df['value']
        time_list = list()
        data_list = list()

        for i in range(len(df)):
            date_time = df_time[i]
            value = df_value[i]
            if time_type == 'hour':
                date_time = date_time.hour
            elif time_type == 'date':
                date_time = date_time.date()
            elif time_type == 'month':
                date_time = datetime.date(date_time.year, date_time.month, 1)
            else:
                date_time = datetime.date(date_time.year, 1, 1)
            if prev_date_time == None or prev_date_time == date_time:
                accumulator_value = accumulator_value + value
            else:
#                 t_val = time_value.strftime('%Y-%m-%d %H:%M:%S') 
                time_list.append(prev_date_time)
                data_list.append(accumulator_value)
                accumulator_value = value
            prev_date_time = date_time
        if prev_date_time != None:
            time_list.append(prev_date_time)
            data_list.append(accumulator_value)
        data = {'time': time_list,
                'value': data_list}
        df = pd.DataFrame(data, columns=['time', 'value'])
        return df
   
    def run_lstm(self, jparams, callback=None):
        '''
        Fetches data from the grid and runs LSTM.
        '''
        jresult = json.loads('{}')
        self.value_key = None
        self.time_attribute = None
        info = None
        error = None
        return_train_data = False
        time_type = 'all'
        grid_path = None
        where_clause = None
        batches = 1
        epochs = 10
        neurons = 10
        use_saved_model = False
        model_name = get_class_name(self)
        if 'info' in jparams:
            info = jparams['info']
        if 'returnTrainData' in jparams:
            return_train_data = jparams['returnTrainData']
        if 'timeType' in jparams:
            time_type = jparams['timeType']
        if 'gridPath' in jparams:
            grid_path = jparams['gridPath']
        if 'where_clause' in jparams:
            where_clause = jparams['where_clause']
        if 'valueKey' in jparams:
            self.value_key = jparams['valueKey']
        if 'timeAttribute' in jparams:
            self.time_attribute = jparams['timeAttribute']
            
        if grid_path == None:
            error = json.loads('{}')
            error['message'] = 'gridPath undefined'
            error['code'] = -1100
        elif self.time_attribute == None:
            error = json.loads('{}')
            error['message'] = 'timeAttribute undefined'
            error['code'] = -1100
            
        if error != None:
            jresult['__error'] = error
            return jresult
        
        if 'batches' in jparams:
            batches = jparams['batches']
        if 'epochs' in jparams:
            epochs = jparams['epochs']
        if 'neurons' in jparams:
            neurons = jparams['neurons']
        if 'useSavedModel' in jparams:
            use_saved_model = jparams['useSavedModel']
        if 'modelName' in jparams:
            model_name = jparams['modelName']
        if callback == None:
            callback = self.value_callback
        data = self.query_map(grid_path, where_clause, callback, self.time_attribute, time_type=time_type)
        df = pd.DataFrame(data, columns=['time', 'value'])
        self.df = self.__prep_data_frame(df, time_type=time_type)

        return self.__run(jresult, use_saved_model, model_name, return_train_data, epochs, neurons, info)
         
    def __forecast(self, test, n_test, n_batch=1, n_lag=1):
        '''
        Returns forecasts and actual values using the specified test dataset.
        '''
        # make forecasts
        forecasts = self.make_forecasts(self.model, test, n_batch, n_lag)
        # inverse transform forecasts and test
        forecasts = self.inverse_transform(self.series, forecasts, self.scaler, n_test+2)
        print("forecasts")
        print(forecasts)
        print()
        print("test")
        print(test)
        
        # invert the transforms on the output part test dataset so that we can correctly
        # calculate the RMSE scores
        actual = [row[n_lag:] for row in test]
        print()
        print(actual)
        actual = self.inverse_transform(self.series, actual, self.scaler, n_test+2)
        return forecasts, actual
        
    def __run(self, jresult, use_saved_model=False, model_name='TemporalLstmDna', 
              return_train_data=False, n_epochs=2, n_neurons=10, info=None, test_values_percent=.2):
        '''
        Runs the LSTM RNN to build the model if use_saved_mode=False or the model does not
        exist. Also, validates the model against the test data split from the dataset.
        '''
        if type(self.df) != DataFrame:
            error = json.loads('{}')
            error['message'] = 'Aborted. Query failed.'
            error['code'] = -1100
            jresult['__error'] = error
            return jresult
        
        if len(self.df['value']) < 5:
            error = json.loads('{}')
            error['message'] = 'Aborted. Not enough samples: ' + str(len(self.df['value']))
            error['code'] = -1101
            jresult['__error'] = error
            return jresult
        
        status = self.create_status(status='started')
        report_start_time = status['Time']
        self.record_status(status)
        
        self.series = self.df.squeeze()
        
                # split data into train and test-sets
        if test_values_percent >= 1 or test_values_percent <= 0:
            test_values_percent = .2
        n_test = int(len(self.series) * test_values_percent)
        self.divider = -n_test
        print("series: %d, test_data: %d" % (len(self.series), n_test))
          
        n_batch = 1      
        n_lag = 1
        n_seq = 3
        self.scaler, self.data_train, self.data_test = self.prepare_data(self.series, n_test, n_lag, n_seq)
       
        # Disable logging steps
        disable_interactive_logging()

        model_found = False
        if use_saved_model:
            # load_model_file() overrides self.scaler created by preapare_data()
            model, scaler = self.load_model_file(model_name)
            if model != None:
                print('Using saved model, scaler, model_info: ' + model_name)
                model_found = True
            else:
                print('Model not found.')
        # Create model if not found
        if model_found == False:
            # fit (train) the model
            print('Fitting (training) model... epochs=%d, neurons=%d' % (n_epochs, n_neurons))
            start_time = time.time()
            # fit model
            self.model = self.fit_lstm(self.data_train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
            print('   took: %f sec' % (time.time() - start_time))
        
            # forecast training data
            # print('forecast the entire training dataset...')
            # start_time = time.time()
            # train_predicted_list, train_expected_list = self.__forecast(self.data_train, n_batch, n_lag)
            # jresult['TrainRmse'] = math.sqrt(mean_squared_error(train_expected_list, train_predicted_list))
            # print('   took: %f sec' % (time.time() - start_time))

        # forecast test data
        start_time = time.time()

        # self.index is not JSON serializable. 
        # Convert its Timestamp items to string values
        time_list = list()
        for ts in self.index[self.divider:]:
            time_list.append(str(ts))
        predicted_list, expected_list = self.__forecast(self.data_test, n_test, n_batch, n_lag)
        # time_list, expected_list, predicted_list = self.forecast_test_dataset()
        print('   took: %f sec' % (time.time() - start_time))
        
        if return_train_data:
            train_data_list = list()
            for value in self.data_raw[0:self.divider]:
                train_data_list.append(value)
            jresult['TrainData'] = train_data_list
            train_time_list = list()
            for ts in self.index[0:self.divider]:
                train_time_list.append(str(ts))
            jresult['TrainTime'] = train_time_list
        jresult['Expected'] = expected_list
        jresult['Predicted'] = predicted_list
        jresult['Time'] = time_list

        # Calculate RMSE
        jresult['Rmse'] = math.sqrt(mean_squared_error(expected_list, predicted_list))
        
        # Store last two (3) values
        last_times = time_list[-3:]
        last_times_str_list = list()
        for i in range(len(last_times)):
            last_times_str_list.append(str(last_times[i]))
        last_values = expected_list[-1]
        self.model_info = {
            "time_type": self.time_type,
            "last_time_list": last_times_str_list,
            "last_value_list": last_values
        }
        
        status = self.create_status(status='done', start_time=report_start_time, jresult=jresult, jinfo=info)
        self.record_status(status)
        self.save_model_file(model_name, use_saved_model)
        return jresult

    def record_status(self, json_status):
        '''
        Records status of DNA.

        This method should be overloaded by the subsclass to record the LSTM run status. 
        It is invoked during LSTM phases to provide real-time status that can be monitored
        by the subclass.
        
        Args:
            status: JSON document containing status attributes
        '''
        key = self.getCurrentTimeKey()
        if self.username != None:
            key = + self.username
        print(key + ": " + repr(json_status))
        return

    def create_status(self, status='started', start_time=None, jresult=None, jinfo=None):
        '''
        Creates status message in JSON to be recorded by the subclass' record_status().
        
        Args:
            status: The following values are recommended:
                'started', 'done', 'in progress', 'warning', 'error', 'failed', 'aborted', 'stopped'
                Default: 'started'
            start_time: The 'StartTime' attribute. If the time it took to complete the
                DNA call is desired, then set this argument with the DNA start time.
            jresult: Optional JSON object containing results.
            jinfo: Optional JSON object containing further information describing the status.
        
        Returns:
            JSON document containing LSTM run status. Use json.dumps() to convert to string
            representation.
        '''
        now = self.getCurrentTime()
        classname = get_class_name(self)
        report = json.loads('{}')
        if classname != None:
            report['Dna'] = classname
        report['Time'] = now
        if start_time != None:
            report['StartTime'] = start_time
        report['Status'] = status
        if jresult != None:
            report['Result'] = jresult
        if jinfo != None:
            report['Info'] = jinfo
        return report
        
    def value_callback(self, key, value):
        '''
        The default value_callback that returns the extracted content from the specified
        value using the specified key. This method can be overwritten if 
        '''
        if type(value) == dict:
            return value[self.value_key];
        else:
            return getattr(value, self.value_key)
        
    def query_map(self, grid_path, where_clause, callback, time_attribute=None, time_type='all'):
        '''
        Abstract method that must be implemented by the subclass.
        
        Queries the temporal list of the specified identity key and returns
        a dictionary containing time-series data with the attributes, 'time'
        and 'value'. 
        
        Args:
            grid_path: Grid path
            callback: The callback function with (key, value) arguments. This
                callback is invoked per record retrieved from the data node.
                The callback must extract the desired column(s) from the JSON value
                object, compute as necessary and return the result that is
                to be used as the raw value. For example, if the value object
                contains 'Quantity' and 'UnitPrice' then the callback may choose
                to return 'Quantity' * 'UnitPrice' for the price value.

        Returns:
            Dictionary with 'time' and 'value' attributes. The 'time' attribute contains
            the time-in-second values, the 'value' attribute contains the observed values.
            Returns None if an error has occurred during query.
        '''
        return None
    
    def transform_to_stationary(self, transformation_type='diff'):
        '''
        Transforms the raw data to stationary data and returns a Series dataset.
        
        Args:
            transformation_type: Transformation type. Currently supports the following types:
                'diff': difference
                
        Returns:
            Transformed Series dataset.
        '''
        if transformation_type == 'diff':
            values = self.__difference(self.df['value'])
        else:
            values = self.__difference(self.df['value'])
        return values
        
    def adfuller(self, series_values=None):
        '''
        Performs a Augmented Dickey-Fuller test on the raw data.
        
        It invokes statsmodels.tsa.stattools.adfuller() with the specified dataset and 
        returns the results from that call.
        
        Args:
            series_values: This argument is typically but not necessarily obtained
                by invoking the transform_to_stationary() method. If unspecified, then
                the raw data is used.
        ''' 
        if type(series_values) != Series:
            values = self.data_raw
        else:
            values = series_values
        return adfuller(values)
      
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        '''
        Frames a time series as a supervised learning dataset.
        Args:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        '''
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    def difference(self, dataset, interval=1):
        '''
        Creates and returns a differenced series.
        '''
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    def prepare_data_train(self, train_series, n_lag, n_seq):
        '''
        Fits and transforms series into train and test sets for supervised learning.
        It invokes scaler.fit_transform() as opposed to scaler.transform()
        
        Args:
            train_series - Train dataset
        '''
        # extract raw values
        raw_values = train_series.values
        # transform data to be stationary
        diff_train_series = self.difference(raw_values, 1)
        diff_values = diff_train_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
        return scaler, supervised.values

    def prepare_data_test(self, scaler, test_series, n_lag, n_seq):
        '''
        Transforms the test series for supervised learning
        It invokes scaler.transform() as opposed to scaler.fit_transform()
        
        Args:
            est_series - Test dataset
        '''
        # extract raw values
        raw_values = test_series.values
        # transform data to be stationary
        diff_test_series = self.difference(raw_values, 1)
        diff_values = diff_test_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaled_values = scaler.transform(diff_values)
        # scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
        return supervised.values
    
    def prepare_data(self, series, n_test, n_lag, n_seq):
        '''
        Transform series into train and test sets for supervised learning. 
        It invokes scaler.fit_transform() as opposed to scaler.transform()

        Args:
            series - Dataset to be split into train and test datasets.
            n_test - Number of test data points. The reshaped data is split into two.
                     The test data with n_test size and the train data
                     with the remaining size. Note that the reshaped data size is smaller
                     than the passed-in series data due to supervised learning data preparation.
        '''
        # extract raw values
        raw_values = series.values
        # transform data to be stationary
        diff_series = self.difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = self.series_to_supervised(scaled_values, n_lag, n_seq)
        supervised_values = supervised.values
        # split into train and test sets
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return scaler, train, test

    def fit_lstm(self, train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
        '''
        Fits an LSTM network to the specified train data.
        
        Returns:
            model - Returns the Squential model.
        '''
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:n_lag], train[:, n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # design network
        model = Sequential()
        model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
        model.add(Dense(y.shape[1]))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        for i in range(nb_epoch):
            model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
            model.reset_states()
        return model
    
    def forecast_lstm(self, model, X, n_batch):
        '''
        Makes one forecast with the specified LSTM model.
        
        Args:
            model - Sequential model
            X - vector containing samples or observed_values.
        
        Returns:
            A list of forecast values.
        '''
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]
    
    def make_forecasts(self, model, test, n_batch, n_lag):
        '''
        Evaluates the persistence model and returns the forecasts.
        '''
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = self.forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts
    
    
    def __forecast_with_saved_model(self, last_observed_series, prepared_data, n_batch=1, n_lag=1):
        
        '''
        Forecasts using the saved model.
        
        Args:
            last_observed_series - Last observed series containing at least last 3 observed
                                    values in sequence 
        Returns:
            forecasts - list of forecasts for the specified last observed series.
        '''
        # make forecasts
        forecasts = self.make_forecasts(self.model, prepared_data, n_batch, n_lag)
        # inverse transform forecasts and test
        forecasts = self.inverse_transform_forecast(last_observed_series, forecasts, self.scaler)
        return forecasts
    
    def make_forecasts_with_saved_model_series(self, model_name, series, n_batch=1, n_lag=1, n_seq=1):
        '''
        Evaluates the specified persistent model. Throws an exception if the model is not found.
        This method uses the saved model and does not update the model info. Hence, it is idempodent.
        
        Returns:
            new_series - It includes the last observed value used to forecast along with the
                         passed-in series values.
            forecasts - A list of forecasted values. The first value is the forecast for the 
                        first value in the passed-in series or the second value of the 
                        returned new_series.
        '''
        self.load_model_file(model_name)
        if self.model == None:
            raise Exception("Specified model file not found [" + model_name + "]")
        if self.scaler == None:
            raise Exception("Specified scaler file not found [" + model_name + "]")
        
        # Insert the last value list in the series
        # Convert string to date list
        last_time_str_list = self.model_info.get("last_time_list")
        last_time_list = list()
        pd_datetime_list = pd.to_datetime(last_time_str_list)
        for i in range(len(pd_datetime_list)):
            if i > 0:
                dt = pd_datetime_list[i].to_pydatetime().date()
                last_time_list.append(dt)
        
        # last value list
        last_value_list = self.model_info.get("last_value_list")
        last_series = Series(index=[last_time_list[-1]], data=[last_value_list[-1]])
        new_series = pd.concat([last_series,series])
        
        prepared_data = self.prepare_data_test(self.scaler, new_series, n_lag, n_seq)
        # prepared_data = self.prepare_data(self.scaler, series, n_lag, n_seq)
        predicted_list = self.__forecast_with_saved_model(new_series, prepared_data, n_batch, n_lag)

        return new_series, predicted_list
    
    
    def make_forecasts_with_saved_model_discrete(self, model_name, observed_date, observed_value, n_batch=1, n_lag=1, n_seq=2, time_type="date"):
        '''
        Evaluates the specified persistent model with the specified discrete value.
        Throws an exception if the model is not found. Unlike make_forecasts_with_saved_model_series(), 
        this method is non-idempodent. It saves the specified observed value for the
        next forecast.
        
        Returns:
            forecast_series - Forecast Series with dates for index and float values as forecasts.
                              The forecast dates start from one unit (day or month) after the
                              specified observed_date. Note that the returned series contains
                              only forecasts. The observed_value is NOT included.

        '''
        self.load_model_file(model_name)
        if self.model == None:
            raise Exception("Specified model file not found [" + model_name + "]")
        if self.scaler == None:
            raise Exception("Specified scaler file not found [" + model_name + "]")
        if self.model_info == None:
            raise Exception("Specified model info file not found [" + model_name + "]")
        
        # Convert string to date list
        last_time_str_list = self.model_info.get("last_time_list")
        last_time_list = list()
        pd_datetime_list = pd.to_datetime(last_time_str_list)
        for i in range(len(pd_datetime_list)):
            if i > 0:
                dt = pd_datetime_list[i].to_pydatetime().date()
                last_time_list.append(dt)
        last_time_list.append(observed_date)
    
        # last value list
        last_value_list = self.model_info.get("last_value_list")
        value_list = last_value_list + [observed_value]
        series = Series(value_list)
        
        prepared_data = self.prepare_data_test(self.scaler, series, n_lag, n_seq)
        predicted_list = self.__forecast_with_saved_model(series, prepared_data, n_batch, n_lag)
        
        forecast_time_list = list()
        # TODO: calculate day_in_sec - 7 days
        day_in_sec = 60*60*24*7
        observed_in_sec = time.mktime(observed_date.timetuple())
        forecast_in_sec = observed_in_sec
        for i in range(len(predicted_list[0])):  
            forecast_in_sec = forecast_in_sec + day_in_sec
            forecast_date = datetime.fromtimestamp(forecast_in_sec).date()
            forecast_time_list.append(forecast_date)
            
        # if time_type == "date":
        #     forecast_in_sec = observed_in_sec + day_in_sec
        # elif time_type == "month":
        #     forecast_in_sec = observed_in_sec + day_in_sec*30
        
        forecast_series = Series(data=predicted_list[0], index=forecast_time_list)
        
        last_times_str_list = list()
        for i in range(len(last_time_list)):
            last_times_str_list.append(str(last_time_list[i]))
        
        self.model_info["last_time_list"] = last_times_str_list
        self.model_info["last_value_list"] = value_list[1:]

        self.save_model_info(model_name)
        return forecast_series
    
    def inverse_difference(self, last_ob, forecast):
        '''
        Inverts differenced forecast.
        
        Returns:
            inverted - Inverted  differecned forecast.
        '''
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted
    
    def inverse_transform(self, series, forecasts, scaler, n_test):
        '''
        Inverses data transform on forecasts. Unlike inverse_transform_forecast(),
        this method expects the specified series to contain the entire set of data
        including train and test data.
        
        Args:
            series - Transformed observed value series
            forecasts - Transformed forecast series 
        Returns:
            inverted - Inverted list of forecasts.
        '''
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = numpy.array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted

    def inverse_transform_forecast(self, series, forecasts, scaler):
        '''          
        Inverses data transform on the specified forecasts. Unlike inverse_transform(),
        this method expects the specified series to contain observed values that map
        directly to the specified forecasts.
        
        Args:
            series - Series containing the last observed values for the specified forecasts.
            forecasts - Transformed forecast series to be inverted.
        Returns:
            Inverted forecast list
        '''
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = numpy.array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            last_ob = series[i]
            inv_diff = self.inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted
     
    def evaluate_forecasts(self, test, forecasts, n_lag, n_seq):
        '''
        Evaluates the RMSE for each forecast time step and prints the RMSE values.
        '''
        for i in range(n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = math.sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))
    
    def __get_model_dir(self):
        '''
        Returns the absolute model directory path. The directory path prepends the working
        directory and it is created if it does not exist. All models should be stored in the
        returned directory path.
        '''
        # make directories first
        # if self.pado == None:
        #     dir_path = 'data/' + self.rpc_context.username
        # else:
        #     dir_path = 'data/' + self.pado.username
        dir_path= self.working_dir + '/data/ml_results'
            
        # Raises an exception if the directory cannot be created.
        os.makedirs(dir_path, exist_ok=True)
        
        return dir_path
        
    def save_model_file(self, model_name, use_saved_model=False):
        '''
        Saves the current model to the specified file path. Prints all the model relevant file names.
        
        Args:
            model_name - The name of the model to be saved.
            use_saved_model - If True, then the model is saved only if the model file
                            does not exist. If False, then the model is saved regardless
                            of whether the model file exists.
        '''
        # serialize model to JSON
        file_path = self.__get_model_dir() + "/" + model_name
        model_json = self.model.to_json()
        with open(file_path + '.json', 'w') as json_file:
            json_file.write(model_json)
            json_file.close()
        # serialize weights to HDF5
        self.model.save_weights(file_path + '.h5')
           
        # Save scaler
        scaler_file_path = file_path + '.scaler'
        joblib.dump(self.scaler, scaler_file_path) 
        
        # Save json model_info
        model_info_filepath = file_path + '-info.json'
        with open(model_info_filepath, 'w') as json_file:
            json.dump(self.model_info, json_file, indent=4)
            json_file.close()
            
        # Duplicate (archive) the model info file. This file can replace the actual model info
        # file if the playback data is run again.
        model_info_archive_filepath = file_path + '-info-archive.json'
        if not use_saved_model or (use_saved_model and not os.path.exists(model_info_archive_filepath)):
            with open(model_info_archive_filepath, 'w') as json_file:
                json.dump(self.model_info, json_file, indent=4)
                json_file.close()        
        
        print("Saved model: " + file_path)
        print("Saved scaler: " + scaler_file_path)
        print("Model-info: " + model_info_filepath)
        
    def load_model_file(self, model_name):
        '''
        Loads the model and its relevant data for the specified model name as follows:
            - model - Sequential model
            - scaler - Scaler used to create model
            - model_info -  Contains last values used to evaluate.
        
        Args:
            model_name - The name of the model to load.
        '''
        # Load json and create model            
        try:
            file_path = self.__get_model_dir() + "/" + model_name
            json_file = open(file_path + '.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(file_path + '.h5')
            self.model = loaded_model
            
            scaler_file_path = file_path + '.scaler'
            self.scaler = joblib.load(scaler_file_path) 
            
            model_info_filepath = file_path + '-info.json'
            with open(model_info_filepath) as json_file:
                self.model_info = json.load(json_file)
                json_file.close()
            
            print("Loaded model from disk: " + file_path)
            print("Loaded scaler from disk: " + scaler_file_path)
            return self.model, self.scaler

        except:
            print("ERROR: Exception occurred while loading model [" + model_name + "]")
            raise
      
    def save_model_info(self, model_name):
        '''
        Saves model_info in the form of JSON representation to the specified file path.
        
        Args:
            model_name - The name of the model.
        '''
        # Save json model_info
        file_path = self.__get_model_dir() + "/" + model_name
        model_info_filepath = file_path + '-info.json'
        with open(model_info_filepath, 'w') as json_file:
            json.dump(self.model_info, json_file, indent=4)
            json_file.close()
                  
    def run_lstm_local(self, grid_path, where_clause=None, time_attribute=None, 
                       value_key=None, use_saved_model=False, model_name='TemporalLstmDna', 
                       callback=None, return_train_data=False, 
                       time_type='all', epochs=2, neurons=10):
        '''
        Runs LSTM RNN locally by fetching data from the grid.
        
        Args:
            time_type - data accumulator by time type. Valid values are
              'all' use all data points without accumulating. Each data point is individually applied in LSTM
              'hour' accumulate data by hour
              'date' accumulate data by date
              'month' accumulate data by month
              'year' accumulate data by year
              Invalid type defaults to 'all'.
        '''
        jparams = json.loads('{}')
        jparams['gridPath'] = grid_path
        jparams['where_clause'] = where_clause
        jparams['returnTrainData'] = return_train_data
        jparams['epochs'] = epochs
        jparams['neurons'] = neurons
        jparams['useSavedModel'] = use_saved_model
        jparams['modelName'] = model_name
        jparams['timeType'] = time_type
        jparams['valueKey'] = value_key
        jparams['timeAttribute'] = time_attribute
        return self.run_lstm(jparams, callback)

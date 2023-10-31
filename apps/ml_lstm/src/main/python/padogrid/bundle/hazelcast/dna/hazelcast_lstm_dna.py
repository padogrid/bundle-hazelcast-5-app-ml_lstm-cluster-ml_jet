"""
Copyright (c) 2023 Netcrest Technologies, LLC. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

'''
Created on July 17, 2023

@author: dpark
'''

import datetime
import logging
import time
import json
import maya

from padogrid.bundle.dna.temporal_lstm_dna import TemporalLstmDna

from hazelcast.predicate import sql
from hazelcast.core import HazelcastJsonValue

logger = logging.getLogger('hazelcast-lstm-dna')

class HazelcastLstmDna(TemporalLstmDna):
    '''
    HazelcastLstmDna implements the following TemporalLstmDna methods.
    
    - query_map(): Returns Hazelcast query results in dictionary containing
                   'time' and 'value' lists. The 'time' list contains times
                   in second, and the 'value' list contains numerical data.
    - record_status(): Invoked repeatedly by TemporalLstmDna during the model
                       generation time to provide the status that can be logged.
    '''

    def __init__(self, feature="stock1-jitter", hazelcast_client=None, username=None, working_dir=None, verbose=0):
        '''
        Constructs a new HazelcastLstmDna object.

        Args:
            hazelcast_client: Hazelcast client instance. This is required if this object is created as
                a non-DNA object, i.e., it is created by a user application.
                    Args:
            working_dir: Working directory path. The model and log files are stored relative
                         to this directory.
        '''
        self.hazelcast_client = hazelcast_client
        self.value_key = feature
        super().__init__(username, working_dir, verbose)

    def get_data(self, result, time_attribute=None, time_type='all'):
        '''
        Creates a dictionary with 'time' and 'value' keys containing the data from the
        specified result set.

        Args:
            result: Data array retrieved from Hazelcast.
            time_attribute (str): Name of the time attribute in the query result set. Default: 'time'
            time_type (str): data accumulator by time type. Valid values are
              'all' use all data points without accumulating. Each data point is individually applied in LSTM
              'hour' accumulate data by hour
              'date' accumulate data by date
              'month' accumulate data by month
              'year' accumulate data by year
              Invalid type defaults to 'all'.

        Returns:
            
        '''
        self.time_type = time_type
        raw_values = list()
        date_list = list()

        print("HazelcastLstmDna.create_data_frame_dna(): dataset length=%d" % len(result))
        if time_attribute == None:
            time_attribute = self.time_attribute
        for entry in result:
            key = entry[0]
            record = entry[1]
            if type(record) == HazelcastJsonValue:
                record = record.loads()
            value = self.value_callback(key, record)
            if value != None:
                if type(record) == dict:
                   time_value = record[time_attribute]
                   panda_time_value = maya.parse(time_value).datetime()
                else:
                    time_value = getattr(record, time_attribute)
                    time_in_sec = time.mktime(time_value)
                    panda_time_value = datetime.fromtimestamp(time_in_sec)
                date_list.append(panda_time_value)
                raw_values.append(value)
        data = {'time': date_list, 'value': raw_values}
        return data

    def query_map(self, grid_path, where_clause, time_attribute=None, time_type='all'):
        '''
        Queries the temporal list of the specified identity key and
        returns a DataFrame object that contains time-series data.

        Args:
            grid_path (str): Grid path
            time_type (str): data accumulator by time type. Valid values are
              'all' use all data points without accumulating. Each data point is individually applied in LSTM
              'hour' accumulate data by hour
              'date' accumulate data by date
              'month' accumulate data by month
              'year' accumulate data by year
              Invalid type defaults to 'all'.

        Returns:
            DataFrame with 'time' and 'value' columns. The 'time' column
            contains the StartValidTime values, the 'value' column contains the column values.
            The returned value is equivalent to TemporalLinearRegression.df.
            Returns None if an error occurred during query.
        '''
        grid_map = self.hazelcast_client.get_map(grid_path)
        if grid_map == None:
            data = None
        else:
            if where_clause == None:
                result = grid_map.entry_set().result()
            else:
                result = grid_map.entry_set(sql(where_clause)).result()
            data = self.get_data(result, time_attribute, time_type)
        return data

    def value_callback(self, key, value):
        '''
        Computes the value expected in the final raw dataset.

        This callback is invoked for every entry from the query result set to
        extract the desired attribute from the value. The value may come in
        different forms and this callback is expected to return the proper
        numerical value. For example, if the value is a JSON object containing
        several attributes, then this callback may choose to return one
        of the attributes or a computed value using one or more attributes.

        Example: If the value object contains 'Quantity' and 'UnitPrice' then
        the callback may choose to return 'Quantity' * 'UnitPrice' for the
        price value.

        Args:
            key: Key
            value: Value

        Returns:
            Data item to use in LSTM. If None, then this value is omitted from
            the raw dataset.
        '''
        if self.value_key != None:
            return value[self.value_key];
        else:
            return value["stock1-jitter"];

    def record_status(self, json_status):
        '''
        Records status of DNA.
        
        Args:
            json_status: JSON document continain DNA status
        '''
        report_map = self.hazelcast_client.get_map('dna_status')
        key = self.getCurrentTimeKey()
        if self.username != None:
            key = "@" + self.username
        report_map.put(key, HazelcastJsonValue(json.dumps(json_status)))
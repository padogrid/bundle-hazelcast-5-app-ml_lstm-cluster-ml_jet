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
Created on July 13, 2023

This module contains the 'transform_list()' function which SimulatorForecastJob invokes
with streamed simulated data to generate forecasts. The 'transform_list()' function expects
an input list with simulated data in a specific format and returns a similarly formatted
forecast values. See details in the function header.

@author: dpark
'''

import logging
import json
import datetime
import os
import pandas as pd
from pandas import Series
from padogrid.bundle.hazelcast.dna.hazelcast_lstm_dna import HazelcastLstmDna

import debugpy

# Set is_debug_enabled to True to enable debugging. Note that the port number
# has been hard coded to 5678. If there are more than one member running
# on the same machine then port conflicts will occur.
is_debug_enabled = False

# Do NOT change this value. It is set automatically if is_debug_enabled is True.
is_debug_initialized = False

# Jet runs from a temporary directory for all Python code executions.
# We want to be able to manage our LSTM models in our own environment so that we
# can reuse them. We do this in the app directory.  
workspace_dir = os.environ['PADOGRID_WORKSPACE']
app_dir = workspace_dir + "/apps/ml_lstm"

logger = logging.getLogger('simulator_forecast')
ch = logging.FileHandler(filename=app_dir + '/log/simulator_forecast.log', encoding='utf-8')
logger.addHandler(ch)
 
def transform_list(input_list):
    """
    Args:
        input_list input list of strings containing observed values for forecasting
        the next set of values. Each item in the list has the following format:
        
        feature|date1;value1:date2;value2...
        
        where date string is in YYYY-MM-dd
        
        Example:
            ['stock1-jitter|2016-11-20;16139:2016-11-26;25729',
             'stock2-jitter|2016-11-20;12345:2016-11-26;35729']
    
    Returns:
        Returns the forecast list containing the observed and forecasted value pairs. The forecasted
        value is always one unit of time ahead of the observed time. For example, the example above
        would return a list similar to the following. The observed value of '2016-11-20' is paired
        with the forecasted value of '2016-11-26'. The unit of time in this case is 7 days (1 week).
        The '^' character is used to pair them.
        
            ['stock1|2016-11-20;16139^2016-11-26;17139:2016-11-26;25729^2016-12-02;25829',
             'stock2|2016-11-20;12345^2016-11-26;12323:2016-11-26;35729^2016-12-02;35221']
            
    Notes:
        Jet's Python function support has the following limitations. 
        
        1. The output list must have the same length as the input list.
        2. Out of index error in the parsing loop even though the string values contain the correct
           number of tokens. 
        3. The input list is constructed by Jet's Python mapping routine which aggregates 
           streamed string values into a list. There is no way to provide a list as an input via
           the Jet API. Only string inputs are allowed.
        4. If the input string contains commas, then the input_list comes in with mangled strings
           and the list cannot be parsed.
    """
    
    t1 = datetime.datetime.now()

    # Eclipse Debugger
    # For PyDev debugging, uncomment the following line. 
    # import pydevd
    # pydevd.settrace("localhost", port=5678)

    # VSCode Debugger
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    global is_debug_enabled, is_debug_initialized
    if is_debug_enabled and is_debug_initialized == False:
        debugpy.listen(5678)
        is_debug_initialized = True
          
    ret_list = []
    
    # Log to a file
    #logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s [%(name)s] %(threadName)s - %(message)s', level=logging.INFO)
    #logging.basicConfig(filename=app_dir + '/log/simulator_forecast.log', encoding='utf-8', level=logging.DEBUG)

    global workspace_dir, app_dir, logger

    logger.info("simulator_forecast.tranform_list() - input_list=" + str(input_list))
    
    try:
    
        # Parse input list and place values into input_dict
        input_dict = {}
        count = 0
        for i in input_list:
            count += 1
            barTokens = i.split("|")
            if len(barTokens) >= 2:
                feature = barTokens[0]
                colonTokens = barTokens[1].split(":")
                str_val = ""
                value_lists = []
                for item in colonTokens:
                    semiTokens = item.split(";")
                    if len(semiTokens) >= 2:
                        date = semiTokens[0]
                        value = float(semiTokens[1])
                        value_lists.append([date, value])
    
                if (len(value_lists) > 0):
                    input_dict[feature] = value_lists   
        
        # Iterate parsed data in input_dict and invoke the forecast routine.
        # Also, build ret_list with forecasted values.
        if len(input_dict) > 0:
            dna = HazelcastLstmDna(working_dir=app_dir)
            for feature, item_list in input_dict.items():
                model_name = "model_" + feature
                forecast_series_list = list()
                for item in item_list:
                    observed_date_str = item[0]
                    observed_date = datetime.datetime.strptime(observed_date_str, "%Y-%m-%d").date() 
                    observed_value = item[1]
                    forecast_series = dna.make_forecasts_with_saved_model_discrete(model_name, observed_date, observed_value)
                    # Insert the observed_value in the series so that we can include it
                    # in the returned list along with the forecasts.
                    observed_series = Series(index=[observed_date], data=[observed_value])
                    new_series = pd.concat([observed_series,forecast_series])
                    forecast_series_list += [new_series]
                str_pairs = ""
                for series in forecast_series_list:
                    single_pair = ""
                    for date, value in series.head(2).items():
                        if len(single_pair) > 0:
                            single_pair += "^"
                        single_pair += str(date) + ";" + str(value)
                    if len(single_pair) > 0:
                        if len(str_pairs) > 0:
                            str_pairs += ":"
                        str_pairs += single_pair
                str_val = feature + "|" + str_pairs
                ret_list.append(str_val)
        
        # If ret_list is not set then return the input list. The returned list must match the
        # same size as the input list (required by Jet).
        if len(ret_list) == 0:
            ret_list = input_list
    
    except Exception as e:
        logger.error("Exception occurred executing simulator_forecast.transform_list [" + str(input_list) + "]")
        logger.error(repr(e))
        raise
    
    t2 = datetime.datetime.now()
    logger.info("simulator_forecast.tranform_list() - ret_list=" + str(ret_list))    
    logger.info("simulator_forecast.tranform_list() - time took (msec)=" + str((t2-t1).total_seconds()*1000))
    return ret_list
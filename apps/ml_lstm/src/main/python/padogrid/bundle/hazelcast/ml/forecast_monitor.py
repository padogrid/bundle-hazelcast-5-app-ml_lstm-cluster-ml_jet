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
Created on July 18, 2023

This application displays a trending chart of observed and forecasted values of LSTM
generated data. It listens on the 'forecast' map for the forecasts generated by
SimulatorJob (Java). The listener displays the received forecasts which also
contains the observed value along with its forecast value.

@author: dpark
'''

import sys
import argparse
from matplotlib import pyplot
import matplotlib.animation as animation
from queue import Queue
import os
import datetime
import time
import hazelcast
from padogrid.bundle.hazelcast.data.PortableFactoryImpl import PortableFactoryImpl

# Disable CPU warning message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class PlotItem:
    def __init__(self, observed_time_list=None, observed_list=None, forecast_time_list=None, forecast_list=None):
        self.observed_time_list = observed_time_list
        self.observed_list = observed_list
        self.forecast_time_list = forecast_time_list
        self.forecast_list = forecast_list

def plot_plot_item(plot_item):
    plot_forecasts(plot_item.observed_time_list, plot_item.observed_list, plot_item.forecast_time_list, plot_item.forecast_list)

def plot_forecasts(observed_time_list, observed_list, forecast_time_list, forecast_list):
    '''
    Plot the specified observed and forecast data. It clears and redraws the plot to
    provide the chart trending effect.
    '''
    ax.clear()
    ax.set(xlabel='Time', ylabel='Value')
    ax.plot(observed_time_list, observed_list, marker='o', label='Observed')
    ax.plot(forecast_time_list, forecast_list, color='red', marker='o', label='Forecast')
    ax.legend(loc = "upper left")
    fig.canvas.draw()
    fig.canvas.flush_events()

def animate(i, interval=500):
    while plot_item_queue.empty() == False:
        plot_item = plot_item_queue.get()
        plot_plot_item(plot_item)

prev_forecast_value = 0

def forecast_received(event):
    '''
    Receives map events from Hazelcast. The events contain ForecastValue objects that are
    readily plotted.
    '''
    global observed_time_list, observed_list, forecast_time_list, forecast_list, max_observed_list_len
    global plot_item_queue
    global prev_forecast_value

    forecast = event.value
    print("forecast_received(): previous_forecast=%f, observed=%f, diff=%f" %
          (prev_forecast_value, forecast.observedValue, forecast.observedValue - prev_forecast_value))
    try:
        observed_date = datetime.datetime(*forecast.observedDate[:6]).date()
        observed_time_list.append(observed_date)
        # observed_time_list.append(forecast.observedDate)
        observed_list.append(forecast.observedValue)

        forecast_date = datetime.datetime(*forecast.forecastDate[:6]).date()
        forecast_time_list.append(forecast_date)
        forecast_list.append(forecast.forecastValue)

        # Limit the list size
        if (len(observed_list)) > max_observed_list_len:
            observed_time_list.pop(0)
            observed_list.pop(0)
        if (len(forecast_list)) > max_observed_list_len:
            forecast_time_list.pop(0)
            forecast_list.pop(0)

        plot_item = PlotItem(observed_time_list, observed_list, forecast_time_list, forecast_list)
        plot_item_queue.put(plot_item)

        #plot_forecasts(observed_time_list, observed_list, forecast_time_list, forecast_list)
    except:
        print("forecast_received() - Exception: ", sys.exc_info()[0])
    prev_forecast_value = forecast.forecastValue

parser = argparse.ArgumentParser(description="Forecast monitor in real time",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-?", action="store_true", help="show this help message and exit")
parser.add_argument("-c", "--cluster", default="ml_jet", help="Hazelcast cluster name")
parser.add_argument("-m", "--map", default="forecast", help="Name of the map that streams forecasted values")
parser.add_argument("-f", "--feature", default="stock1-jitter", help="Name of the feature to monitor. This feature is extracted from the forecast map in real time.");

args = vars(parser.parse_args())

# '-?' in addition to '-h', '--help'
is_help = args["?"]
if is_help:
    parser.print_help()
    exit()

cluster_name = args["cluster"]
grid_path = args["map"]
feature = args["feature"]

# The max number of observed/forecast data points to display.    
max_observed_list_len = 30

# lists to be populated with data for plotting
observed_time_list = list()
observed_list = list()
forecast_time_list = list()
forecast_list = list()

# PlotItem queue for chart animation
plot_item_queue = Queue()

# Login
client = hazelcast.HazelcastClient(cluster_name=cluster_name,
                                    cluster_members=[
                                            "localhost:5701",
                                            "localhost:5702",
                                            "localhost:5703"
                                        ],
                                    lifecycle_listeners=[
                                        lambda state: print("Hazelcast Lifecycle: ", state),
                                    ],
                                    portable_factories=PortableFactoryImpl.factories())

# Listen on the forecast map for forecasts generated via Jet
forecast_map = client.get_map(grid_path)
forecast_map.add_entry_listener(key=feature, include_value=True, added_func=forecast_received, updated_func=forecast_received)
print("--------------------------------------")
print("      cluster: %s" % (cluster_name))
print("          map: %s" % (forecast_map.name))
print("feature (key): %s" % (feature))
print("--------------------------------------")
print("The trending chart will not show until it receives data from the map, '%s'." % (forecast_map.name))

# Build up the observed_list before plotting
while len(observed_list) < 2:
    time.sleep(1.0)

# Label and plot
fig, ax = pyplot.subplots(1, figsize=(14, 8))
fig.suptitle(feature + " LSTM Forecast")
ax.set(xlabel='Time', ylabel='Value')
ax.plot(observed_time_list, observed_list, marker='o')

# Animate
anim = animation.FuncAnimation(fig, animate, interval=200)

# Display and block
pyplot.gcf().canvas.manager.set_window_title("PadoGrid LSTM Forecast")
pyplot.show()

# Shutdown Hazelcast client
client.shutdown()

# For debugging
#while True:
#    time.sleep(1.0)

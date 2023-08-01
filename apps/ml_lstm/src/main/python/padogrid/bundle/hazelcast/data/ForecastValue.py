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
 
"""
ForecastValue contains the previous observed value and the forecast value.
This class is NOT generated.
"""

from hazelcast.serialization.api import Portable
import time
from padogrid.bundle.hazelcast.data import PortableFactoryImpl

class ForecastValue(Portable):
    def __init__(self, id=None, observedDate=None, observedValue=None, forecastDate=None, forecastValue=None,):
        self.id=id
        self.observedDate=observedDate
        self.observedValue=observedValue
        self.forecastDate=forecastDate
        self.forecastValue=forecastValue
        
    def get_class_id(self):
        return PortableFactoryImpl.ForecastValue_CLASS_ID

    def get_factory_id(self):
        return PortableFactoryImpl.FACTORY_ID

    def get_class_version(self):
        return 1

    def write_portable(self, writer):
        writer.write_string("id", self.id)
        if self.observedDate == None: 
            writer.write_long("observedDate", -1)
        else:
            writer.write_long("observedDate", self.observedDate)
        writer.write_double("observedValue", self.observedValue)
        if self.forecastDate == None: 
            writer.write_long("forecastDate", -1)
        else:
            writer.write_long("forecastDate", self.forecastDate)
        writer.write_double("forecastValue", self.forecastValue)

    def read_portable(self, reader):
        self.id = reader.read_string("id")
        val = reader.read_long("observedDate")
        if val == -1:
            self.observedDate = None
        else:
            self.observedDate = time.localtime(val/1000)
        self.observedValue = reader.read_double("observedValue")
        val = reader.read_long("forecastDate")
        if val == -1:
            self.forecastDate = None
        else:
            self.forecastDate = time.localtime(val/1000)
        self.forecastValue = reader.read_double("forecastValue")
    
    def __repr__(self):
        return "[id=" + repr(self.id) \
            + ", observedDate=" + repr(self.observedDate) \
            + ", observedValue=" + repr(self.observedValue) \
            + ", forecastDate=" + repr(self.forecastDate) \
            + ", forecastValue=" + repr(self.forecastValue) + "]"
    

    def __str__(self):
        return "[id=" + str(self.id) \
            + ", observedDate=" + str(self.observedDate) \
            + ", observedValue=" + str(self.observedValue) \
            + ", forecastDate=" + str(self.forecastDate) \
            + ", forecastValue=" + str(self.forecastValue) + "]"
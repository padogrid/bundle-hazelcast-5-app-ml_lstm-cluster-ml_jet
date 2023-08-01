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
ForexRecord is generated code. To modify this class, you must follow the
guidelines below.

  - Always add new fields and do NOT delete old fields.
  - If new fields have been added, then make sure to increment the version number.

@generator com.netcrest.pado.tools.hazelcast.VersionedPortableClassGenerator
@schema ForexRecordsCountry.schema
@date Tue May 18 10:20:44 EDT 2021
"""

from hazelcast.serialization.api import Portable
import time
from padogrid.bundle.hazelcast.data import PortableFactoryImpl

class ForexRecord(Portable):
    def __init__(self, timestamp=None, bidOpen=None, bidHigh=None, bidLow=None,
                bidClose=None, bidVolume=None, askOpen=None, askHigh=None,
                askLow=None, askClose=None, askVolume=None, nextAvgUp=None):
        self.timestamp = timestamp;
        self.bidOpen = bidOpen;
        self.bidHigh = bidHigh;
        self.bidLow = bidLow;
        self.bidClose= bidClose;
        self.bidVolume = bidVolume;
        self.askOpen = askOpen;
        self.askHigh = askHigh;
        self.askLow = askLow;
        self.askClose = askClose;
        self.askVolume = askVolume;
        self.nextAvgUp = nextAvgUp;
       
    def get_class_id(self):
        return PortableFactoryImpl.ForexRecord_CLASS_ID

    def get_factory_id(self):
        return PortableFactoryImpl.FACTORY_ID

    def get_class_version(self):
        return 1

    def write_portable(self, writer):
        if self.timestamp == None: 
            writer.write_long("timestamp", -1)
        else:
            writer.write_long("timestamp", self.timestamp)
        writer.write_double("bidOpen", self.bidOpen)
        writer.write_double("bidHigh", self.bidHigh)
        writer.write_double("bidLow", self.bidLow)
        writer.write_double("bidClose", self.bidClose)
        writer.write_double("bidVolume", self.bidVolume)
        writer.write_double("askOpen", self.askOpen)
        writer.write_double("askHigh", self.askHigh)
        writer.write_double("askLow", self.askLow)
        writer.write_double("askClose", self.askClose)
        writer.write_double("askVolume", self.askVolume)
        writer.write_string("nextAvgUp", self.nextAvgUp)

    def read_portable(self, reader):
        val = reader.read_long("timestamp")
        if val == -1:
            self.timestamp = None
        else:
            self.timestamp = time.localtime(val/1000)
        self.bidOpen = reader.read_double("bidOpen")
        self.bidHigh = reader.read_double("bidHigh")
        self.bidLow = reader.read_double("bidLow")
        self.bidClose = reader.read_double("bidClose")
        self.bidVolume = reader.read_double("bidVolume")
        self.askOpen = reader.read_double("askOpen")
        self.askHigh = reader.read_double("askHigh")
        self.askLow = reader.read_double("askLow")
        self.askClose = reader.read_double("askClose")
        self.askVolume = reader.read_double("askVolume")
        self.nextAvgUp = reader.read_string("nextAvgUp")
    
    def __repr__(self):
        return "[timestamp=" + repr(self.timestamp) \
              + ", bidOpen=" + repr(self.bidOpen) \
              + ", bidHigh=" + repr(self.bidHigh) \
              + ", bidLow=" + repr(self.bidLow) \
              + ", bidClose=" + repr(self.bidClose) \
              + ", bidVolume=" + repr(self.bidVolume) \
              + ", askOpen=" + repr(self.askOpen) \
              + ", askHigh=" + repr(self.askHigh) \
              + ", askLow=" + repr(self.askLow) \
              + ", askClose=" + repr(self.askClose) \
              + ", askVolume=" + repr(self.askVolume) \
              + ", nextAvgUp=" + repr(self.nextAvgUp) + "]"
    

    def __str__(self):
        return "[timestamp=" + str(self.timestamp) \
              + ", bidOpen=" +  str(self.bidOpen) \
              + ", bidHigh=" +  str(self.bidHigh) \
              + ", bidLow=" +  str(self.bidLow) \
              + ", bidClose=" +  str(self.bidClose) \
              + ", bidVolume=" +  str(self.bidVolume) \
              + ", askOpen=" +  str(self.askOpen) \
              + ", askHigh=" +  str(self.askHigh) \
              + ", askLow=" +  str(self.askLow) \
              + ", askClose=" +  str(self.askClose) \
              + ", askVolume=" +  str(self.askVolume) \
              + ", nextAvgUp=" + str(self.nextAvgUp) + "]"
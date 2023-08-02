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
Created on May 20, 2021

@author: dpark
'''

from datetime import datetime
import os

class Dna():
    '''
    A parent class of all DNA classes.
    
    All DNA classes must inherit this class and provide the following attributes:
    
        username: User name mainly used to store reports in the report/dna map. If not specified,
                  then the OS login user name is used. If the login user name is not available
                  then it defaults to the user name 'dna'.
    '''
    
    pado = None
    
    def __init__(self, username=None):
        self.username = username
        if self.username == None:
            self.username = os.getlogin()
        if self.username == None:
            self.username = 'dna'
    
    def getCurrentTime(self):
        '''
        Returns the current time in string form.
        '''
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def getCurrentTimeKey(self):
        '''
        Returns the current time to use as key to Hazelcast map.
        '''
        return datetime.now().strftime('%Y%m%d%H%M%S')
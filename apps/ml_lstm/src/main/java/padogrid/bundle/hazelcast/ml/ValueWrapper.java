/*
 * Copyright (c) 2023 Netcrest Technologies, LLC. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package padogrid.bundle.hazelcast.ml;

import java.io.Serializable;

import com.hazelcast.shaded.org.json.JSONObject;

public class ValueWrapper implements Serializable {
    private static final long serialVersionUID = 1L;
    
	String feature;
    String timeStr;
    double value;

    public ValueWrapper() {}

    public ValueWrapper(String feature, JSONObject json) {
        this.feature = feature;
        this.timeStr = json.getString("time");    
        this.value = json.getDouble(feature);    
    }

    public ValueWrapper(String feature, String timeStr, double value) {
        this.feature = feature;
        this.timeStr = timeStr;
        this.value = value;
    }

    public String getFeature() {
        return feature;
    }

    public void setFeature(String feature) {
        this.feature = feature;
    }

    public String getTimeStr() {
        return timeStr;
    }

    public void setTimeStr(String timeStr) {
        this.timeStr = timeStr;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}

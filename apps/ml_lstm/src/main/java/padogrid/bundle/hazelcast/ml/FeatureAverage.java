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
import java.time.LocalDate;

/**
 * {@linkplain FeatureAverage} accumulates the feature values and returns the average value.
 * 
 * @author dpark
 *
 */
public class FeatureAverage implements Serializable {

	private static final long serialVersionUID = 1L;

	private String feature;
	private LocalDate date;
	private double value;
	private long updateTime;
	private int count = 0;

	public FeatureAverage() {

	}

	public FeatureAverage(String feature, LocalDate date, double value) {
		this.feature = feature;
		this.date = date;
		setValue(value);
	}

	public String getFeature() {
		return feature;
	}

	public LocalDate getDate() {
		return date;
	}

	public double getValue() {
		if (count == 0) {
			return value;
		}
		return value/count;
	}

	public void setValue(double value) {
		this.value = value;
		setUpdateTime(System.currentTimeMillis());
		count = 1;
	}

	public long getUpdateTime() {
		return updateTime;
	}

	public void setUpdateTime(long updatedTime) {
		this.updateTime = updatedTime;
	}

	public void add(double value) {
		this.value += value;
		setUpdateTime(System.currentTimeMillis());
		count++;
	}

	@Override
	public String toString() {
		return "FeatureAverage [feature=" + feature + ", date=" + date + ", value=" + value
				+ ", updateTime=" + updateTime + "]";
	}
}

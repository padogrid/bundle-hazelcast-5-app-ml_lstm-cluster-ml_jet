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
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.Map;
import java.util.concurrent.ConcurrentSkipListMap;

import com.hazelcast.core.HazelcastJsonValue;
import com.hazelcast.shaded.org.json.JSONArray;
import com.hazelcast.shaded.org.json.JSONException;
import com.hazelcast.shaded.org.json.JSONObject;

/**
 * {@linkplain SimulatorAccumulator} accumulates feature values for each day.
 * 
 * @author dpark
 *
 */
public class SimulatorAccumulator implements Serializable {

	private static final long serialVersionUID = 1L;

	/**
	 * {@linkplain #lagInMsec} is used to delay before
	 * committing the final feature values. {@linkplain SimulatorAccumulator} delays
	 * this amount of time since the last
	 * update before committing the feature values. Note that this logic cannot
	 * handle the last day since {@linkplain #toListStr()}, which commits the
	 * feature
	 * values, * will not be invoked until the next update becomes available.
	 */
	private int lagInMsec = 1000;

	private String feature;
	private ConcurrentSkipListMap<LocalDate, FeatureAverage> dateStockAverageMap = new ConcurrentSkipListMap<LocalDate, FeatureAverage>();

	private SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");

	public SimulatorAccumulator(String feature, int lagInMsec) {
		this.feature = feature;
		this.lagInMsec = lagInMsec;
	}

	public SimulatorAccumulator(String feature) {
		this.feature = feature;
	}

	private LocalDate createYearMonthDate(String time) throws ParseException {
		return Instant.ofEpochMilli(dateFormatter.parse(time).getTime()).atZone(ZoneId.systemDefault()).toLocalDate();
	}

	public void add(ValueWrapper wrapper) {
		try {
			LocalDate date = createYearMonthDate(wrapper.getTimeStr());
			FeatureAverage sa = dateStockAverageMap.get(date);
			if (sa == null) {
				sa = new FeatureAverage(feature, date, wrapper.getValue());
				dateStockAverageMap.put(date, sa);
			} else {
				sa.add(wrapper.getValue());
			}
		} catch (ParseException e) {
			e.printStackTrace();
		}
	}

	public void add(String timeStr, double value) {
		try {
			LocalDate date = createYearMonthDate(timeStr);
			FeatureAverage sa = dateStockAverageMap.get(date);
			if (sa == null) {
				sa = new FeatureAverage(feature, date, value);
				dateStockAverageMap.put(date, sa);
			} else {
				sa.add(value);
			}
		} catch (ParseException e) {
			e.printStackTrace();
		}
	}

	public void add(HazelcastJsonValue jv) {
		JSONObject json = new JSONObject(jv);
		try {
			LocalDate date = createYearMonthDate(json.getString("time"));
			FeatureAverage sa = dateStockAverageMap.get(date);
			if (sa == null) {
				sa = new FeatureAverage(feature, date, json.getDouble(feature));
				dateStockAverageMap.put(date, sa);
			} else {
				sa.add(json.getDouble(feature));
			}
		} catch (JSONException | ParseException e) {
			e.printStackTrace();
		}
	}

	public String feature() {
		return feature;
	}

	public Map<LocalDate, FeatureAverage> getDateStockAverageMap() {
		return dateStockAverageMap;
	}

	/**
	 * Returns the lag in msec.
	 * {@linkplain SimulatorAccumulator} delays
	 * this amount of time since the last
	 * update before committing the feature values. Note that this logic cannot
	 * handle the last day since {@linkplain #toListStr()}, which commits the
	 * feature
	 * values, will not be invoked until the next update becomes available.
	 */
	public int getLagInMsec() {
		return this.lagInMsec;
	}

	/**
	 * Returns a list of accumulated average values in the following format:
	 * <p>
	 * 
	 * <pre>
	 * feature|date1;value1:date2;value2:...
	 * </pre>
	 * <p>
	 * where
	 * <p>
	 * <ul>
	 * <li>feature is the feature name</li>
	 * <li>date has the format of yyyy-MM-dd</li>
	 * <li>value is a double value</li>
	 * </ul>
	 * <p>
	 * 
	 * @return null if there are no accumulated values.
	 * 
	 * @example stock1-jitter|2023-07-24;79.34:2023-07-25;80.01
	 */
	public String toListStr() {
		Map<LocalDate, FeatureAverage> map = getDateStockAverageMap();
		long now = System.currentTimeMillis();
		String strList = "";
		for (Map.Entry<LocalDate, FeatureAverage> e : map.entrySet()) {
			LocalDate date = e.getKey();
			FeatureAverage sa = e.getValue();
			if (now - sa.getUpdateTime() > lagInMsec) {
				if (strList.length() > 0) {
					strList += ":";
				}
				strList += sa.getDate() + ";" + sa.getValue();
				map.remove(date);
			}
		}
		if (strList.length() > 0) {
			strList = feature + "|" + strList;
		} else {
			strList = null;
		}
		return strList;
	}

	/**
	 * Returns a JSON string representation that contains the accumulated stock
	 * average values. The returned value has the following format:
	 * <p>
	 * 
	 * <pre>
	 * {
	 *    "feature": "stock1"
	 *    "items": [ { "date": "2024-07-24", "value": 79.34}, 
	 *               { "date": "2024-07-25", "value": 80.01}, ... ]
	 * }
	 * </pre>
	 * 
	 * @return null if there no accumulated values.
	 */
	public String toJsonStr() {
		long now = System.currentTimeMillis();
		Map<LocalDate, FeatureAverage> map = getDateStockAverageMap();
		JSONArray items = new JSONArray();
		for (Map.Entry<LocalDate, FeatureAverage> e : map.entrySet()) {
			LocalDate date = e.getKey();
			FeatureAverage sa = e.getValue();
			if (now - sa.getUpdateTime() > lagInMsec) {
				JSONObject sumJson = new JSONObject();
				sumJson.put("date", sa.getDate().toString());
				sumJson.put("value", sa.getValue());
				items.put(sumJson);
				map.remove(date);
			}
		}
		if (items.length() == 0) {
			return null;
		} else {
			JSONObject json = new JSONObject();
			json.put("feature", this.feature);
			json.put("items", items);
			return json.toString();
		}
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((feature == null) ? 0 : feature.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		SimulatorAccumulator other = (SimulatorAccumulator) obj;
		if (feature == null) {
			if (other.feature != null)
				return false;
		} else if (!feature.equals(other.feature))
			return false;
		return true;
	}

	@Override
	public String toString() {
		return dateStockAverageMap.toString();
	}
}

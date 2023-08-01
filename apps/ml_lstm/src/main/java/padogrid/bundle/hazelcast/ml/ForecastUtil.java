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

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;

import padogrid.bundle.hazelcast.data.ForecastValue;

/**
 * ForecastUtil provides utility methods for parsing forecast values. 
 * @author dpark
 *
 */
public class ForecastUtil {

	/**
	 * Parses the specified forecast string and returns an array of ForecastValue
	 * objects. It returns an empty array if the specified forecast string is null,
	 * empty, or invalid.
	 * 
	 * @param forecasts Forecast string value
	 *                  <p>
	 * Example:
	 * 
	 *                  <pre>
	 * Greece|2017-01-21;-22892.544563293457^2017-01-22;-23009.375022888184
	 *                  </pre>
	 * 
	 * @return An array of ForecastValue objects extracted from the specified
	 *         forecast string.
	 * @throws ParseException Thrown if value is not a numeric value.
	 */
	public static List<ForecastValue>parseForecasts(String forecasts) throws ParseException {

		if (forecasts == null || forecasts.length() == 0) {
			return Collections.emptyList();
		}
		String[] split = forecasts.split("\\|");
		if (split.length < 2) {
			return Collections.emptyList();
		}

		ArrayList<ForecastValue> list = new ArrayList<ForecastValue>(10);
		SimpleDateFormat fomatter = new SimpleDateFormat("yyyy-MM-dd");
		String id = split[0];
		String rest = split[1];
		split = rest.split(":");
		for (String pair : split) {
			String[] split2 = pair.split("\\^");
			if (split2.length == 2) {
				ForecastValue f = new ForecastValue();
				f.setId(id);
				int count = 0;
				for (String item : split2) {
					count++;
					String[] split3 = item.split(";");
					if (split3.length == 2) {
						String dateStr = split3[0];
						String valueStr = split3[1];
						Date date = fomatter.parse(dateStr);
						double value = Double.parseDouble(valueStr);
						if (count == 1) {
							f.setObservedDate(date);
							f.setObservedValue(value);
						} else {
							f.setForecastDate(date);
							f.setForecastValue(value);
						}
					}
				}
				if (count == 2) {
					list.add(f);
				}
			}
		}
		return list;
	}
}

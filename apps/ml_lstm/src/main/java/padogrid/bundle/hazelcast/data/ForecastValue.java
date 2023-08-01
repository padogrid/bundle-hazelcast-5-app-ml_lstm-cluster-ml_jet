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
package padogrid.bundle.hazelcast.data;

import java.io.IOException;
import java.util.Date;

import com.hazelcast.nio.serialization.PortableReader;
import com.hazelcast.nio.serialization.PortableWriter;
import com.hazelcast.nio.serialization.VersionedPortable;

/**
 * ForecastValue contains observed value and its forecast value.
 * 
 * @author dpark
 *
 */
public class ForecastValue implements VersionedPortable
{
	private String id;
	private Date observedDate;
	private double observedValue;
	private Date forecastDate;
	private double forecastValue;


	public ForecastValue() {}
	
	public ForecastValue(String id, Date observedDate, double observedValue, Date forecastDate, double forecastValue) {
		this.id = id;
		this.observedDate = observedDate;
		this.observedValue = observedValue;
		this.forecastDate = forecastDate;
		this.forecastValue = forecastValue;
	}
	
	public String getId() {
		return id;
	}

	public void setId(String id) {
		this.id = id;
	}

	public Date getObservedDate() {
		return observedDate;
	}

	public void setObservedDate(Date observedDate) {
		this.observedDate = observedDate;
	}

	public double getObservedValue() {
		return observedValue;
	}

	public void setObservedValue(double observedValue) {
		this.observedValue = observedValue;
	}

	public Date getForecastDate() {
		return forecastDate;
	}

	public void setForecastDate(Date forecastDate) {
		this.forecastDate = forecastDate;
	}

	public double getForecastValue() {
		return forecastValue;
	}

	public void setForecastValue(double forecastValue) {
		this.forecastValue = forecastValue;
	}

	@Override
	public int getClassId() 
	{
		return PortableFactoryImpl.ForecastValue_CLASS_ID;
	}

	@Override
	public int getFactoryId() {
		return PortableFactoryImpl.FACTORY_ID;
	}
	
	@Override
	public int getClassVersion() {
		return 1;
	}

	@Override
	public void writePortable(PortableWriter writer) throws IOException {
		writer.writeString("id", id);
		if (this.observedDate == null) {
			writer.writeLong("observedDate", -1L);
		} else {
			writer.writeLong("observedDate", this.observedDate.getTime());
		}
		writer.writeDouble("observedValue", observedValue);
		if (this.forecastDate == null) {
			writer.writeLong("forecastDate", -1L);
		} else {
			writer.writeLong("forecastDate", this.forecastDate.getTime());
		}
		writer.writeDouble("forecastValue", forecastValue);
	}

	@Override
	public void readPortable(PortableReader reader) throws IOException {
		id = reader.readString("id");
		long l = reader.readLong("observedDate");
		if (l != -1L) {
			this.observedDate = new Date(l);
		}
		this.observedValue = reader.readDouble("observedValue");
		l = reader.readLong("forecastDate");
		if (l != -1L) {
			this.forecastDate = new Date(l);
		}
		this.forecastValue = reader.readDouble("forecastValue");
	}

	@Override
	public String toString() {
		return "ForecastValue [id=" + id + ", observedDate=" + observedDate + ", observedValue=" + observedValue
				+ ", forecastDate=" + forecastDate + ", forecastValue=" + forecastValue + "]";
	}
}

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
  * This code has been manually modified.
  *
  * ForexRecord is generated code. To modify this class, you must follow the
  * guidelines below.
  * <ul>
  * <li>Always add new fields and do NOT delete old fields.</li>
  * <li>If new fields have been added, then make sure to increment the version number.</li>
  * </ul>
  *
  * @generator com.netcrest.pado.tools.hazelcast.VersionedPortableClassGenerator
  * @schema FOREX_eurusd-minute-High.schema
  * @date Fri Jul 09 14:04:48 EDT 2021
**/
public class ForexRecord implements VersionedPortable
{
	private Date timestamp;
	private double bidOpen;
	private double bidHigh;
	private double bidLow;
	private double bidClose;
	private double bidVolume;
	private double askOpen;
	private double askHigh;
	private double askLow;
	private double askClose;
	private double askVolume;
	private String nextAvgUp;

	public ForexRecord()
	{
	}

	public void setTimestamp(Date timestamp) {
		this.timestamp=timestamp;
	}

	public Date getTimestamp() {
		return this.timestamp;
	}

	public void setBidOpen(double bidOpen) {
		this.bidOpen=bidOpen;
	}

	public double getBidOpen() {
		return this.bidOpen;
	}

	public void setBidHigh(double bidHigh) {
		this.bidHigh=bidHigh;
	}

	public double getBidHigh() {
		return this.bidHigh;
	}

	public void setBidLow(double bidLow) {
		this.bidLow=bidLow;
	}

	public double getBidLow() {
		return this.bidLow;
	}

	public void setBidClose(double bidClose) {
		this.bidClose=bidClose;
	}

	public double getBidClose() {
		return this.bidClose;
	}

	public void setBidVolume(double bidVolume) {
		this.bidVolume=bidVolume;
	}

	public double getBidVolume() {
		return this.bidVolume;
	}

	public void setAskOpen(double askOpen) {
		this.askOpen=askOpen;
	}

	public double getAskOpen() {
		return this.askOpen;
	}

	public void setAskHigh(double askHigh) {
		this.askHigh=askHigh;
	}

	public double getAskHigh() {
		return this.askHigh;
	}

	public void setAskLow(double askLow) {
		this.askLow=askLow;
	}

	public double getAskLow() {
		return this.askLow;
	}

	public void setAskClose(double askClose) {
		this.askClose=askClose;
	}

	public double getAskClose() {
		return this.askClose;
	}

	public void setAskVolume(double askVolume) {
		this.askVolume=askVolume;
	}

	public double getAskVolume() {
		return this.askVolume;
	}

	public void setNextAvgUp(String nextAvgUp) {
		this.nextAvgUp=nextAvgUp;
	}

	public String getNextAvgUp() {
		return this.nextAvgUp;
	}


	@Override
	public int getClassId() 
	{
		return PortableFactoryImpl.ForexRecord_CLASS_ID;
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
		if (this.timestamp == null) {
			writer.writeLong("timestamp", -1L);
		} else {
			writer.writeLong("timestamp", this.timestamp.getTime());
		}
		writer.writeDouble("bidOpen", bidOpen);
		writer.writeDouble("bidHigh", bidHigh);
		writer.writeDouble("bidLow", bidLow);
		writer.writeDouble("bidClose", bidClose);
		writer.writeDouble("bidVolume", bidVolume);
		writer.writeDouble("askOpen", askOpen);
		writer.writeDouble("askHigh", askHigh);
		writer.writeDouble("askLow", askLow);
		writer.writeDouble("askClose", askClose);
		writer.writeDouble("askVolume", askVolume);
		writer.writeString("nextAvgUp", nextAvgUp);
	}

	@Override
	public void readPortable(PortableReader reader) throws IOException {
		long l = reader.readLong("timestamp");
		if (l != -1L) {
			this.timestamp = new Date(l);
		}
		this.bidOpen = reader.readDouble("bidOpen");
		this.bidHigh = reader.readDouble("bidHigh");
		this.bidLow = reader.readDouble("bidLow");
		this.bidClose = reader.readDouble("bidClose");
		this.bidVolume = reader.readDouble("bidVolume");
		this.askOpen = reader.readDouble("askOpen");
		this.askHigh = reader.readDouble("askHigh");
		this.askLow = reader.readDouble("askLow");
		this.askClose = reader.readDouble("askClose");
		this.askVolume = reader.readDouble("askVolume");
		this.nextAvgUp = reader.readString("nextAvgUp");
	}
    
	@Override
	public String toString()
	{
		return "[askClose=" + this.askClose
			 + ", askHigh=" + this.askHigh
			 + ", askLow=" + this.askLow
			 + ", askOpen=" + this.askOpen
			 + ", askVolume=" + this.askVolume
			 + ", bidClose=" + this.bidClose
			 + ", bidHigh=" + this.bidHigh
			 + ", bidLow=" + this.bidLow
			 + ", bidOpen=" + this.bidOpen
			 + ", bidVolume=" + this.bidVolume
			 + ", nextAvgUp=" + this.nextAvgUp
			 + ", timestamp=" + this.timestamp + "]";
	}
}

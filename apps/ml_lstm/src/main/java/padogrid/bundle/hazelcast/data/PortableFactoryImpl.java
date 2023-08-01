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

import com.hazelcast.nio.serialization.Portable;
import com.hazelcast.nio.serialization.PortableFactory;

/**
 * <b>NOTE:</b> This class has been manually updated to include ForecastValue.
 * <p>
 * PortableFactoryImpl is generated code. To manually modified the code, make
 * sure to follow the same naming conventions. Otherwise, the code generator may
 * not work.
 * 
 * @generator com.netcrest.pado.tools.hazelcast.VersionedPortableClassGenerator
 * @date Sun Jul 23 06:02:14 EDT 2023
 */
public class PortableFactoryImpl implements PortableFactory {
	public static final int FACTORY_ID = Integer.getInteger("padogrid.demo.hazelcast.data.PortableFactoryImpl.factoryId",
			30002);

	static final int __FIRST_CLASS_ID = Integer.getInteger("padogrid.demo.hazelcast.data.PortableFactoryImpl.firstClassId",
			1002);
	static final int ForexRecord_CLASS_ID = __FIRST_CLASS_ID;
	static final int ForecastValue_CLASS_ID = ForexRecord_CLASS_ID + 1;

	public Portable create(int classId) {
		if (classId == ForexRecord_CLASS_ID) {
			return new ForexRecord();
		} else if (classId == ForecastValue_CLASS_ID) {
			return new ForecastValue();
		} else {
			return null;
		}
	}
}

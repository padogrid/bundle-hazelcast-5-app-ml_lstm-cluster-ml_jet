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
package padogrid.mqtt.connectors;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.UUID;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.eclipse.paho.mqttv5.client.MqttClient;
import org.eclipse.paho.mqttv5.common.MqttException;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastJsonValue;
import com.hazelcast.map.IMap;
import com.hazelcast.shaded.org.json.JSONObject;

import padogrid.mqtt.client.cluster.HaMqttClient;
import padogrid.mqtt.client.cluster.IHaMqttConnectorSubscriber;

/**
 * {@linkplain HazelcastJsonConnector} bridges MQTT topics to Hazelcast maps. It
 * converts MQTT topic names to Hazelcast map names by replacing '/' with '_'
 * (underscore). The following plugin properties are supported.
 * 
 * <ul>
 * <li>topicFilters - Comma separated list of topic filters. Default: #</li>
 * <li>keyType - SEQUENCE, UUID, TIME, KEY. Default: SEQUENCE</li>
 * <li>hazelcast.client.config - Hazelcast client configuration file path. Required property.</li>
 * </ul>
 */
public class HazelcastJsonConnector implements IHaMqttConnectorSubscriber {

    enum KeyType {
        SEQUENCE, UUID, TIME, KEY;

        public static KeyType getKeyType(String value) {
            if (value.equalsIgnoreCase(SEQUENCE.name())) {
                return SEQUENCE;
            } else if (value.equalsIgnoreCase(UUID.name())) {
                return UUID;
            } else if (value.equalsIgnoreCase(TIME.name())) {
                return TIME;
            } else if (value.equalsIgnoreCase(KEY.name())) {
                return KEY;
            } else {
                return SEQUENCE;
            }
        }
    }

    private Logger logger = LogManager.getLogger(HazelcastJsonConnector.class);

    private String pluginName;
    private String description;
    private Properties props;

    private String topicFilters;
    private String key;
    private KeyType keyType;

    private HazelcastInstance hzInstance;
    private HaMqttClient haclient;

    private SimpleDateFormat dateFormatter = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ");
    private Map<String, Long> sequenceKeyMap = new HashMap<String, Long>();

    @Override
    public boolean init(String pluginName, String description, Properties props, String... args) {
        logger.info(String.format("Initializing %s...", HazelcastJsonConnector.class.getSimpleName()));
        this.pluginName = pluginName;
        this.description = description;
        this.props = props;

        topicFilters = props.getProperty("topicFilters", "#");
        key = props.getProperty("key");
        String keyTypeStr = props.getProperty("keyType", "SEQUENCE");
        keyType = KeyType.getKeyType(keyTypeStr);
        String hazelcastConfigFile = props.getProperty("hazelcast.client.config");
        if (hazelcastConfigFile != null) {
            System.setProperty("hazelcast.client.config", hazelcastConfigFile);
        }

        logger.info("hazelcast.client.config=" + System.getProperty("hazelcast.client.config"));
        try {
            hzInstance = HazelcastClient.getOrCreateHazelcastClient();
        } catch (Exception ex) {
            logger.error(String.format(
                    "Error occurred while creating Hazelcast client instance [hazelcast.client.config=%s]",
                    hazelcastConfigFile));
            throw ex;
        }
        logger.info("Initialized.");
        return true;
    }

    @Override
    public void stop() {
        if (hzInstance != null) {
            hzInstance.shutdown();
        }
    }

    @Override
    public void start(HaMqttClient haclient) {
        try {
            String[] split = topicFilters.split(",");
            for (String tf : split) {
                haclient.subscribe(tf.trim(), 0);
            }
            logger.info(String.format("Started [topicFilters=%s].", topicFilters));
        } catch (MqttException e) {
            stop();
            logger.error(String.format("Error occurred while subscribing to MQTT virtual cluster. %s discarded.",
                    HazelcastJsonConnector.class.getSimpleName()), e);
        }
    }

    @Override
    public void messageArrived(MqttClient client, String topic, byte[] payload) {
        String json_doc = new String(payload);
        JSONObject json = new JSONObject(json_doc);
        HazelcastJsonValue hjson = new HazelcastJsonValue(json_doc);
        String mapName = topic.replaceAll("/", "_");
        String mapKey;
        final IMap<String, HazelcastJsonValue> map = hzInstance.getMap(mapName);
        switch (keyType) {
            case KEY:
                Object keyObj = null;
                if (this.key != null) {
                    keyObj = json.get(this.key);
                }
                if (keyObj == null) {
                    Long seqNum = sequenceKeyMap.getOrDefault(mapName, 0L);
                    mapKey = Long.toString(++seqNum);
                } else {
                    mapKey = keyObj.toString();
                }
                break;
            case TIME:
                mapKey = dateFormatter.format(new Date());
                break;
            case UUID:
                mapKey = UUID.randomUUID().toString();
                break;
            case SEQUENCE:
            default:
                Long seqNum = sequenceKeyMap.getOrDefault(mapName, 0L);
                mapKey = Long.toString(++seqNum);
                sequenceKeyMap.put(mapName, seqNum);
                break;
        }
        map.set(mapKey, hjson);
    }
}
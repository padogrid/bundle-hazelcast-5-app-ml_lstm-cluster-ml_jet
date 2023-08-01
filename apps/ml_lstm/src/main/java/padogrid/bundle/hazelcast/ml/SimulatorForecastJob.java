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

import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastJsonValue;
import com.hazelcast.jet.JetService;
import com.hazelcast.jet.config.JobConfig;
import com.hazelcast.jet.pipeline.JournalInitialPosition;
import com.hazelcast.jet.pipeline.Pipeline;
import com.hazelcast.jet.pipeline.Sink;
import com.hazelcast.jet.pipeline.SinkBuilder;
import com.hazelcast.jet.pipeline.Sources;
import com.hazelcast.jet.python.PythonServiceConfig;
import com.hazelcast.jet.python.PythonTransforms;
import com.hazelcast.shaded.org.json.JSONObject;

import padogrid.bundle.hazelcast.data.ForecastValue;

/**
 * SimulatorForecastJob accumulates "time" values streamed from the
 * "journal" event journal map and applies them to the LSTM model
 * to obtain the next set of forecast values.
 * 
 * @author dpark
 *
 */
public class SimulatorForecastJob {

	final static String forecastMapName = "forecast";

	public static void main(String[] args) {
		//System.setProperty("hz.jet.resource-upload-enabled", "true");
		Options options = new Options();
		Option helpOpt = new Option("h", "help", false, "Print this message");
		helpOpt.setRequired(false);
		options.addOption(helpOpt);
		Option help2Opt = new Option("?", null, false, "Print this message");
		help2Opt.setRequired(false);
		options.addOption(help2Opt);
		Option featureOpt = new Option("f", "feature", true, "Feature name. Default: stock1-jitter");
		featureOpt.setRequired(false);
		options.addOption(featureOpt);
		Option observedMapOpt = new Option("j", "journal", true,
				"Name of the journal map that streams observed values. Default: journal");
		observedMapOpt.setRequired(false);
		options.addOption(observedMapOpt);
		CommandLineParser parser = new DefaultParser();
		HelpFormatter formatter = new HelpFormatter();
		CommandLine cmd = null;// not a good practice, it serves its purpose

		boolean isHelp = false;
		boolean isHelp2 = false;
		String featureValue = null;
		String journalMapValue = null;
		try {
			cmd = parser.parse(options, args);
			isHelp = cmd.hasOption("help");
			isHelp2 = cmd.hasOption("?");
			featureValue = cmd.getOptionValue("feature", "stock1-jitter");
			journalMapValue = cmd.getOptionValue("journal", "journal");
		} catch (ParseException e) {
			System.out.println(e.getMessage());
			formatter.printHelp("SimulatorForecastJob", options);
			System.exit(1);
		}

		if (isHelp || isHelp2) {
			formatter.printHelp("SimulatorForecastJob", options);	
			System.exit(0);
		}

		final String feature = featureValue;
		final String observedMapName = journalMapValue;

		System.out.println();
		System.out.println("--------------------------------------------------------------------");
		System.out.printf("Submitting: %s%n", SimulatorForecastJob.class.getSimpleName());
		System.out.printf("feature=%s, observedMapName=%s, forecastMapName=%s%n", feature, observedMapName, forecastMapName);
		System.out.println("--------------------------------------------------------------------");
		System.out.println();

		Pipeline p = Pipeline.create();
		p.readFrom(Sources.<String, HazelcastJsonValue>mapJournal(observedMapName,
				JournalInitialPosition.START_FROM_CURRENT))
				.withIngestionTimestamps()
				.map(e -> new ValueWrapper(feature, new JSONObject(e.getValue().getValue())))
				// .groupingKey(e -> new JSONObject(e.getValue().getValue()).get(feature))
				.mapStateful(() -> new SimulatorAccumulator(feature), (accumulator, wrapper) -> {
					accumulator.add(wrapper);
					return accumulator;
				})
				.map(accumulator -> {
					String inputStr = accumulator.toListStr();
					if (inputStr != null && inputStr.length() > 0) {
						// Log inputs (for debugging)
						System.out.print("Accumulated Input: " + inputStr + " ==> ");
					}
					return inputStr;
				})
				.apply(PythonTransforms.mapUsingPython(
						new PythonServiceConfig().setBaseDir("src/main/python")
								.setHandlerModule("padogrid.bundle.hazelcast.ml.jet_forecast")))
				.setLocalParallelism(1)
				.writeTo(MySinks.build());

		JobConfig cfg = new JobConfig().setName("simulator-forecast-" + feature).addClass(SimulatorForecastJob.class);

		JetService jet = Hazelcast.bootstrappedInstance().getJet();
		jet.newJob(p, cfg);
	}

	static class MySinks {
		static Sink<Object> build() {
			return SinkBuilder.sinkBuilder("LSTM Forecast", ctx -> ctx.hazelcastInstance().getMap(forecastMapName))
					.receiveFn((forecastMap, item) -> {
						if (item.toString().length() > 0) {
							System.out.println("Sink: " + item.getClass() + " - " + item);
							try {
								List<ForecastValue> forecasts = ForecastUtil.parseForecasts(item.toString());
								for (ForecastValue value : forecasts) {
									forecastMap.put(value.getId(), value);
								}
								System.out.println(forecasts);
							} catch (Exception ex) {
								ex.printStackTrace();
							}
						}
					}).build();
		}
	}
}

package com.ml.hw2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataInputer {

	public static DataSet getDataFromFile(String dataFile, String featureFile) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		List<Feature> features = new ArrayList<Feature>();
		
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			features = FeatureInputer.getFeaturesList(featureFile);
			dataset = new DataSet(features.size() - 1, features);
			
			while (true) {
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);

				double[] featureValues = new double[values.length];

				for (int i = 0; i < values.length; i++) {
					
					featureValues[i] = Double.parseDouble(values[i]);
					
				}

				Data data = new Data(featureValues);
				dataset.addData(data);

			}
			
		} finally {
			readFile.close();
		}
		
		return dataset;
	}


	
}

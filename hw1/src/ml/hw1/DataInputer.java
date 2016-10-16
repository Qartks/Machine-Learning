package ml.hw1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DataInputer {
	
	private static double[] featureValueMean;
	private static double[] featureMin;
	private static double[] featureMax;

	public static DataSet getDataFromFile(String dataFile, String featureFile, boolean normalizeData, boolean isTrainingData) throws Exception {
		BufferedReader readFile = null;
		DataSet dataset = null;
		double[] featureValueSum;
		List<Feature> features = new ArrayList<Feature>();
		
		try {
			readFile = new BufferedReader(new FileReader(dataFile));
			features = FeatureInputer.getFeaturesList(featureFile);
			dataset = new DataSet(features.size() - 1, features);

			featureValueSum = new double[features.size()];
			
			featureMin = new double[features.size()];
			featureMax = new double[features.size()];
			
			for(int i = 0; i<features.size(); i++){
				featureMin[i] = Double.POSITIVE_INFINITY;
				featureMax[i] = Double.NEGATIVE_INFINITY;
			}
			
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
					featureValueSum[i] += featureValues[i];
					if(featureValues[i] > featureMax[i]){
						featureMax[i] = featureValues[i];
					}
					if(featureValues[i] < featureMin[i]){
						featureMin[i] = featureValues[i];
					}
				}

				Data data = new Data(featureValues);
				dataset.addData(data);

			}
		} finally {
			readFile.close();
		}
		if(isTrainingData && normalizeData) {
			featureValueMean = new double[features.size()];
			for (int i = 0; i < features.size() - 1; i++) {
				featureValueMean[i] = featureValueSum[i] / dataset.dataSize();
			}
		}
		return normalizeData ? normalizeDataSandS(dataset) : dataset;
	}

	private static DataSet normalizeDataSandS(DataSet dataset) {
		
		for(Data d : dataset.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				if( i != dataset.getLabelIndex()){
					featureValues[i] = (featureValues[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
				}
			}
		}
		
		return dataset;
		
	}

	private static DataSet normalizeData(DataSet dataset) {
		
		double[] featureVariance = new double[featureValueMean.length];
		for(Data d : dataset.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				featureVariance[i] += Math.pow(featureValues[i] - featureValueMean[i],2);
			}

		}
		
		for(int i = 0; i< featureVariance.length; i++){
			featureVariance[i] = Math.sqrt(featureVariance[i]);
		}
		
		for(Data d : dataset.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				if( i != dataset.getLabelIndex()){
					featureValues[i] = (featureValues[i] - featureValueMean[i]) / featureVariance[i];
				}
			}
		}
		
		return dataset;
	}
	
	
	public static void writeToFile(DataSet dataset, String fileName) throws FileNotFoundException{
		
		PrintWriter pw = new PrintWriter(fileName);
		
		for(Data d : dataset.getData()){
			pw.write(Arrays.toString(d.getFeatureValues()));
			pw.write("\n");
		}
		
		pw.close();
	}

}

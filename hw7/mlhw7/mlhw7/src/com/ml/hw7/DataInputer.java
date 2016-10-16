package com.ml.hw7;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class DataInputer {
	
	static int noOFFeatures = 200;
	
	public static DataSet getData(String fileName) throws Exception{
		
		BufferedReader readFile = null;
		DataSet dataSet = new DataSet(noOFFeatures);
		try {
			
			readFile = new BufferedReader(new FileReader(fileName));
			
			while(true){
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				
				Data d = new Data(noOFFeatures);
				
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);
				
				double labelVal = Double.parseDouble(values[values.length -1]);
				d.setLabelValue(labelVal);
				for(int j = 0; j < values.length - 1; j++){
					if(values[j].equals("nan")){
						d.setFeatureValueAtIndex(j, Double.NaN);
					} else {
						double tempVal = Double.parseDouble(values[j]);
						d.setFeatureValueAtIndex(j, tempVal);
					}
				}
				
				d.setDataSet(dataSet);
				dataSet.addData(d);
			}
		} finally {
			
			readFile.close();
			
		}
		return dataSet;
	}
	

	@SuppressWarnings("resource")
	public static DataSet getPollutedData(String fileName, String fileName2, int featureCount) throws Exception {
		
		BufferedReader readFile = null;
		BufferedReader readLabelFile = null;
		DataSet dataSet = new DataSet(featureCount);
		try {
			
			readFile = new BufferedReader(new FileReader(fileName));
			readLabelFile = new BufferedReader(new FileReader(fileName2));
			
			while(true){
				String line = readFile.readLine();
				String line2 = readLabelFile.readLine();
				if (line == null || line2 == null) {
					break;
				}

				if (line.trim().length() == 0 || line2.trim().length() == 0) {
					break;
				}
				
				Data d = new Data(featureCount);
				
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);
				String[] values2 = line2.trim().split(delims);
				
				double labelVal = Double.parseDouble(values2[0]);
				d.setLabelValue(labelVal);
				
				for(int j = 0; j < values.length - 1; j++){
					double tempVal = Double.parseDouble(values[j]);
					d.setFeatureValueAtIndex(j, tempVal);
				}
				
				d.setDataSet(dataSet);
				dataSet.addData(d);
			}
		} finally {
			
			readFile.close();
			
		}
		return dataSet;
	}
	
	public static DataSet getEMData(String fileName) throws Exception{
		
		BufferedReader readFile = null;
		DataSet dataSet = new DataSet(noOFFeatures);
		try {
			
			readFile = new BufferedReader(new FileReader(fileName));
			
			while(true){
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				
				Data d = new Data(noOFFeatures);
				
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);
				
				double labelVal = Double.parseDouble(values[values.length -1]);
				d.setLabelValue(labelVal);
				for(int j = 0; j < values.length; j++){
					double tempVal = Double.parseDouble(values[j]);
					d.setFeatureValueAtIndex(j, tempVal);
				}
				
				d.setDataSet(dataSet);
				dataSet.addData(d);
			}
		} finally {
			
			readFile.close();
			
		}
		return dataSet;
	}

	public static FeatureStats getFeatureStatsFromFile(String fileName) throws Exception {
		
		BufferedReader readFile = null;
		FeatureStats fStat = new FeatureStats(noOFFeatures+1);
		int k = 0;
		try{
			readFile = new BufferedReader(new FileReader(fileName));
			
			while(true){
				String line = readFile.readLine();
				
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				
				String delim = "\\s+|,";
				String[] values = line.trim().split(delim);
				
				for(int i = 1; i< values.length; i++){
					double val = Double.parseDouble(values[i]);
					fStat.setValue(k, i-1, val);
				}
				k++;
			}
		} finally {
			readFile.close();
		}
		
		return fStat;
	}

	public static DataSet getECOCData(String fName) throws IOException {
		BufferedReader readFile = null;
		
		DataSet dataSet = new DataSet(1754);
		
		try {
			readFile = new BufferedReader(new FileReader(fName));
			
			while(true){
				String line = readFile.readLine();
				
				if(line == null){
					break;
				}
				
				if(line.trim().length() == 0){
					break;
				}
				
				String delim = "\\s+|,";
				String[] values = line.split(delim);
				
				Data d = new Data(1754);
				
				double labelVal = Double.parseDouble(values[0]);
				d.setLabelValue(labelVal);
				
				for(int i = 1; i < values.length; i++){
					String[] feat_Val = values[i].split(":");
					d.setFeatureValueAtIndex(Integer.parseInt(feat_Val[0]), Double.parseDouble(feat_Val[1]));
				}
				
				d.setDataSet(dataSet);
				dataSet.addData(d);
			}
			
		} finally {
			readFile.close();
		}
		
		return dataSet;
	}

	public static DataSet getUCIData(String dir, String file) throws Exception {
		String dataPath = dir + file + ".data";
		String configPath = dir + file +  ".config";
		
		BufferedReader readDataFile = null;
		BufferedReader readConfigFile = null;
		
		DataSet dataSet = null;
		HashMap<Integer, List<String>> featureProp = new HashMap<Integer, List<String>>();
		HashMap<Integer, Integer> featureMap = new HashMap<Integer, Integer>();
		try {
			readConfigFile = new BufferedReader(new FileReader(configPath));
			readDataFile = new BufferedReader(new FileReader(dataPath));
			
			String line = readConfigFile.readLine();
//			if(line == null || line.trim().length() == 0){
//				break;
//			}
			String delim = "\\s+|,";
			String[] counts = line.split(delim);
			
			int dataSize = Integer.parseInt(counts[0]);
			int noOfCategoricalfeature = Integer.parseInt(counts[1]);
			int noOfContinuosfeature = Integer.parseInt(counts[2]);
			
			int totalFeatures = 0;
			for(int i = 0; i < noOfCategoricalfeature + noOfContinuosfeature; i++){
				line = readConfigFile.readLine();
				featureMap.put(i, totalFeatures);
				if(line.equals("-1000")){
//					featureProp.put(i, null);
					totalFeatures++;
				} else {
					String[] vals = line.split(delim);
					int catFeatSize = Integer.parseInt(vals[0]);
					List<String> list = new ArrayList<String>(catFeatSize);
					for(int j = 1; j < vals.length; j++){
						list.add(vals[j]);
					}
					totalFeatures += (list.size());
					Collections.sort(list);
					featureProp.put(i, list);
				}
			}
			
			String[] label = readConfigFile.readLine().split(delim);
			List<String> labels = new ArrayList<String>();
			for(int i = 1; i< label.length - 1; i++){
				labels.add(label[i]);
			}
			Collections.sort(labels);
			dataSet = new DataSet(totalFeatures);
			
			while(true){
				Data d = new Data(totalFeatures);
				List<Integer> missingValues = new ArrayList<Integer>();
				String line2 = readDataFile.readLine();
				
				if(line2 == null || line2.trim().length() == 0){
					break;
				}
				
//				if(line2.contains("?")){
//					continue;
//				}
				String[] vals = line2.split(delim);
				
				int f = 0;
				
				String l = vals[vals.length - 1];
//				if(l.equals("d")){
//					d.setLabelValue(1);
//				} else {
//					d.setLabelValue(0);
//				}
				for(String ll : labels){
					if(l.equals(ll)){
						d.setLabelValue(1);
					} else {
						d.setLabelValue(0);
					}
				}
				for(int j = 0; j < vals.length - 1; j++){
					String val = vals[j];
					if(val.trim().equals("?")){
						missingValues.add(j);
						if(featureProp.containsKey(j)){
							List<String> list = featureProp.get(j);
							for(String s : list){
								f++;
							}
						} else {
							f++;
						}
						continue;
					}
					if(featureProp.containsKey(j)){
						List<String> list = featureProp.get(j);
						for(String s : list){
							if(val.equals(s)){
								d.setFeatureValueAtIndex(f, 1);
							}
							f++;
						}
//						f+=(list.size() * 2);
					} else {
						d.setFeatureValueAtIndex(f, Double.parseDouble(val));
						f++;
					}
				}
				d.missingValues = missingValues;
				d.setDataSet(dataSet);
				dataSet.addData(d);
			}
			
		} finally {
			readConfigFile.close();
			readDataFile.close();
		}
		
		dataSet.featureProp = featureProp;
		dataSet.featureMap = featureMap;
		return dataSet;
	}
	
	public static void writeDataToFile(String fileLocation, DataSet dataSet) throws Exception {
		File file = new File(fileLocation);
		FileWriter fw = new FileWriter(file.getAbsoluteFile());
		BufferedWriter bw = new BufferedWriter(fw);
		for(Data data : dataSet.getData()) {
			bw.write(data.toString()+"\n");
		}
		bw.close();
	}

}

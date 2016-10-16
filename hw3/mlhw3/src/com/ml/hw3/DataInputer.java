package com.ml.hw3;

import java.io.BufferedReader;
import java.io.FileReader;

public class DataInputer {
	

	static int noOfData = 4601;
	static int noOFFeatures = 57;
	
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

}

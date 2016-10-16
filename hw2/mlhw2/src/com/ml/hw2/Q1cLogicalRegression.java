package com.ml.hw2;

import java.util.Collections;

public class Q1cLogicalRegression {

	public static void main(String[] args) throws Exception {

		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data", 
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names");
		
//		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt", 
//				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt");
		
		Normalize.normalizeDataSandS(dataSet);
//		Normalize.normalizeData(dataSet);
		
		double lambda = 0.001;
		double threshold = 0.001;
//		GDImplementation.learn(dataSet, lambda, false, threshold);
		
		errorWithKFold(10, dataSet, lambda);
		
	}
	
private static void errorWithKFold(int k, DataSet dataSet, double lambda) throws Exception {
		
		int dataPerFold = dataSet.dataSize()/ k;
		
		Collections.shuffle(dataSet.getData());
		for(int fold=0; fold< k ; fold++) {			
			
			DataSet trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			
			for(int x = 0; x < dataSet.dataSize(); x++) {
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(dataSet.getData().get(x));
				} else {
					trainingData.addData(dataSet.getData().get(x));
				}
			}
			
			System.out.println("Fold "+ (fold+1));
			GradDesImplement.learn(dataSet, lambda, true, 0, true, testData);
			
		}
	}

}

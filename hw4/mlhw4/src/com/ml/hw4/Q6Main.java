package com.ml.hw4;

import java.util.Collections;

public class Q6Main {

	public static void main(String[] args) throws Exception {
		
		DataSet2 dataSet = OldDataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names", false, false);
		
		errorWithKFold(10, dataSet);
		
	}
	
private static void errorWithKFold(int k, DataSet2 dataSet) throws Exception {
		
		double avgError = 0;
		double avgPerError = 0;
		int dataPerFold = dataSet.dataSize()/ k;
		
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold< k ; fold++) {
			
			DataSet2 trainingData = new DataSet2(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet2 testData = new DataSet2(dataSet.getLabelIndex(), dataSet.getFeatures());
			
			
			for(int x = 0; x < dataSet.dataSize(); x++) {
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(dataSet.getData().get(x));
				} else {
					trainingData.addData(dataSet.getData().get(x));
				}
			}
			BaggingImplementation bg = new BaggingImplementation(trainingData);
			bg.train(50);
			double errorOnFold = 1 - bg.test(testData);
			avgPerError += errorOnFold;
			System.out.println("Accuracy on Fold "+ (fold+1) + " -> " + errorOnFold);
		}
		System.out.println("\nAverage Accuracy Percentage :"+ avgPerError/k);
	
	}

}

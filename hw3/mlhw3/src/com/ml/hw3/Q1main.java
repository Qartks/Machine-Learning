package com.ml.hw3;

import java.util.Collections;

public class Q1main {

	public static void main(String[] args) throws Exception {
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		errorWithKFold(allData, 10);
	}

	private static void errorWithKFold(DataSet dataSet, int k) {
		
		double avgTestError = 0;
		double avgTrainError = 0;
		int dataPerFold = dataSet.getDataSize()/ k;
		GDAImplementation gda = new GDAImplementation();
		
		
		for(int fold=0; fold< k ; fold++) {
			Collections.shuffle(dataSet.getData());
			double error = 0;
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			for(int x = 0; x < dataSet.getDataSize(); x++) {
				Data d = dataSet.getData().get(x);
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(d);
					d.setDataSet(testData);
				} else {
					trainingData.addData(d);
					d.setDataSet(trainingData);
				}
			}
			
			GDAModel model = gda.train(trainingData);
				
			error = gda.test(model, trainingData);
			avgTrainError += error;
			System.out.println("Training Error For Fold " + (fold+1) + " -> " + error);
			
			error = gda.test(model, testData);
			avgTestError += error;
			System.out.println("Testing Error For Fold " + (fold+1) + " -> " + error);
			System.out.println();
		}
		
		System.out.println();
		System.out.println("Average Training Error -> " + avgTrainError/k);
		System.out.println("Average Testing Error -> " + avgTestError/k);
		
	}

}

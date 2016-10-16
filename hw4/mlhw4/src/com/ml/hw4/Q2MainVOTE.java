package com.ml.hw4;

import java.util.Collections;

public class Q2MainVOTE {

	public static void main(String[] args) throws Exception {
		
		DataSet allData = DataInputer.getUCIData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/vote/","vote");
		
		allData.computeFeatureStats();
		allData.calculateMissingValues();
		
		percentSelection(allData);
		System.out.println("");
		System.out.println("");
		errorWithKFold(allData, 10);
		
		System.out.println("Hi");
	}
	
	private static void percentSelection(DataSet allData) throws Exception {
		
		double[] pert = {5, 10, 15, 20, 30, 50, 80};
		
		for(int k = 0; k < pert.length; k++){
			int size = allData.getDataSize();
			int c = (int) (0.01 * pert[k] * size);
			
			DataSet T = new DataSet(allData.getFeatureSize());
			DataSet X = new DataSet(allData.getFeatureSize());
			for(int i = 0; i< size; i++){
				Data d = allData.getData().get(i);
				if( i < c){				
					T.addData(d);
				} else {
					X.addData(d);
				}
			}
			
			AdaBoostImplementation adaB = new AdaBoostImplementation(T, 100);
			adaB.train(T, X, 100, true);
			System.out.println("Percentage:" + pert[k] + " -> Training Error:" + adaB.trainErr.error + " -> Testing Error:" + adaB.aL.error);
		}
	}

	private static void errorWithKFold(DataSet dataSet, int k) throws CloneNotSupportedException {
		
		int dataPerFold = dataSet.getDataSize()/ k;
		
		
		double avgError = 0;
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold< k ; fold++) {
			
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
			AdaBoostImplementation adaB = new AdaBoostImplementation(trainingData, 100);
			
			adaB.train(trainingData, testData, 100, true);
			System.out.println("Fold " + (fold+1) + " -> " + (1 - adaB.aL.error));
			avgError += adaB.aL.error;
		}
		
		System.out.println(1 - (avgError / k));
		
	}


}

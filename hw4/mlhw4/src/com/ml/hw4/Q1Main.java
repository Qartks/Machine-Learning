package com.ml.hw4;

import java.util.Collections;


public class Q1Main {

	public static void main(String[] args) throws Exception {
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		
		errorWithKFold(allData, 10);
		
	}
	
private static void errorWithKFold(DataSet dataSet, int k) throws CloneNotSupportedException {
		
		int dataPerFold = dataSet.getDataSize()/ k;
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
			AdaBoostImplementation adaB = new AdaBoostImplementation(trainingData, 1000);
			
			adaB.train(trainingData, testData, 1000, false);
			
			if(fold == 0){
				break;
			}
		}
		
	}

}

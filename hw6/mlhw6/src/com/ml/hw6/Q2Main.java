package com.ml.hw6;

import java.util.Arrays;
import java.util.Collections;

public class Q2Main {

	public static void main(String[] args) throws Exception {
		
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		
		Normalize.normalizeDataSandS(allData);		
		
		Collections.shuffle(allData.getData());
		int size = allData.getDataSize();
		int c = (int) (0.50 * size);
		
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
		

//		Normalize.normalizeData(trainingData, testData);
//		SMO smo = new SMO(1, 0.001, 0.01, allData, 20);
//		smo.buildLagrangeMultipliers();
//		System.out.println(smo.predict(trai));
//		System.out.println(smo.predict(X));
		errorKFold(10, allData);
		

	}
	
	private static void errorKFold(int k, DataSet dataSet) throws Exception {
		
		int dataPerFold = dataSet.getDataSize()/ k;
		Collections.shuffle(dataSet.getData());
		double avgTrainAcc = 0;
		double avgTestAcc = 0;
		
		
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
			System.out.println("Fold " + (fold + 1) + ":");
			SMO smo = new SMO(1, 0.001, 0.01, trainingData, 20);
			smo.buildLagrangeMultipliers();
			avgTrainAcc += smo.predict(trainingData);
			avgTestAcc += smo.predict(testData);
		}
		
		System.out.println("\n\nAverage Training Accuracy: " + (avgTrainAcc/k));
		System.out.println("Average Testing Accuracy: " + (avgTestAcc/k));
		
	}

}

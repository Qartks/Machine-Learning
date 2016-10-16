package com.ml.hw7;

import java.util.Collections;

public class Q1MainA {

	public static void main(String[] args) throws Exception {
		
		DataSet allData = DataInputer.getData("C:/Users/eesha_000/Downloads/spambase.data");
		
		errorWithKFold(10, allData, 3);
		
//		DataSet imageData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_train2.txt");
		
//		Normalize.normalizeDataSandS(allData);
		
//		Collections.shuffle(allData.getData());
//		int size = allData.getDataSize();
//		int c = (int) (0.70 * size);
//		
//		DataSet T = new DataSet(allData.getFeatureSize());
//		DataSet X = new DataSet(allData.getFeatureSize());
//		for(int i = 0; i< size; i++){
//			Data d = allData.getData().get(i);
//			if( i < c){				
//				T.addData(d);
//			} else {
//				X.addData(d);
//			}
//		}
		
//		Kernel kernel = new EuclidDistance(T.getDataSize());
//		KNNImplementation knn = new KNNImplementation(7, 0, false, T,kernel);
//		knn.execute(X);
		
		
		
	}

	private static void errorWithKFold(int k, DataSet dataSet, int n) throws Exception {
		int dataPerFold = dataSet.getDataSize()/ k;
		double avgTestAcc = -1;
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
			System.out.println("Fold " + (fold + 1) + ":");
//			Kernel kernel = new GaussianKernel(0, 2);
			Kernel kernel = new EuclidDistance(0);
//			Normalize.normalizeDataSandS(trainingData, testData);
			KNNImplementation knn = new KNNImplementation(n, 0, false, trainingData, kernel);
			avgTestAcc += knn.execute(testData);
		}
		System.out.println("Average Testing Accuracy: " + (1 - avgTestAcc/k)*100);
		
	}

}

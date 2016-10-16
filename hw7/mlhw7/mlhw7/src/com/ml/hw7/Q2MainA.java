package com.ml.hw7;

import java.util.Collections;

public class Q2MainA {
	
	public static void main(String[] args) throws Exception {
		
//		forSpam();
		forDigits();
		
	}

	private static void forDigits() throws Exception {
		DataSet trainData = DataInputer.getData("C:/Users/eesha_000/Downloads/ML/Q3_image_train.txt");
		DataSet testData = DataInputer.getData("C:/Users/eesha_000/Downloads/ML/Q3_image_test.txt");
		int size = trainData.getDataSize();
		
		Kernel kernel = new CosineDistance(size);
		KNNImplementation knn = new KNNImplementation(1, 0.83, true, trainData, kernel);
		double error = knn.execute(testData);
		System.out.println("Accuracy -> " + error);
	}

	private static void forSpam() throws Exception{
		
		DataSet allData = DataInputer.getData("C:/Users/eesha_000/Downloads/spambase.data");
		errorWithKFold(10, allData);
	}
	
	private static void errorWithKFold(int k, DataSet dataSet) throws Exception {
		int dataPerFold = dataSet.getDataSize()/ k;
		Collections.shuffle(dataSet.getData());
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
			KNNImplementation knn = new KNNImplementation(1, .5, true, trainingData, new EuclidDistance(trainingData.getDataSize()));
			avgTestAcc += knn.execute(testData);
		}
		System.out.println("Average Testing Accuracy: " + (avgTestAcc/k));
		
	}

}

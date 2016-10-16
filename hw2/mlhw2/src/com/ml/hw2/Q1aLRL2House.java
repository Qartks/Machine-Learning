package com.ml.hw2;

import org.ejml.data.DenseMatrix64F;

public class Q1aLRL2House {

	public static void main(String[] args) throws Exception {
		
		DataSet trainSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt", 
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt");
		
		DataSet testSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_test.txt", 
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt");
		
//		Normalize.normalizeData(trainSet, testSet);
//		Normalize.normalizeDataSandS(trainSet, testSet);
		
		int trainDataSize = trainSet.getData().size();
		int trainFeatureSize = trainSet.getFeatures().size();
		double[] trainFeatureMatrix = trainSet.getFeatureMatrix();
		double[] trainLabelMatrix = trainSet.getLabelMatrix();
		double lambda = 0.01d;
		double avgTrainingError = 0;
		double avgTestingError = 0;
		
//		for(int i = 0; i < 100; i++){
		DenseMatrix64F weightMatrix = LRImplementation.train(
				trainDataSize,
				trainFeatureSize,
				trainFeatureMatrix,
				trainLabelMatrix,
				lambda);

		double trainingError = LRImplementation.calculateError(weightMatrix,
				trainDataSize,
				trainFeatureSize,
				trainFeatureMatrix,
				trainLabelMatrix);
		
		avgTrainingError += trainingError;
		System.out.println("Error on Training Set -> " + trainingError);
		
		int testDataSize = testSet.getData().size();
		int testFeatureSize = testSet.getFeatures().size();
		double[] testFeatureMatrix = testSet.getFeatureMatrix();
		double[] testLabelMatrix = testSet.getLabelMatrix();

		double testingError = LRImplementation.calculateError(weightMatrix,
				testDataSize,
				testFeatureSize,
				testFeatureMatrix,
				testLabelMatrix);
		
		avgTestingError += testingError;
		System.out.println("Error on Testing Set -> " + testingError);
//		}
		
	}

}

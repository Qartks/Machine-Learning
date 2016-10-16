package com.ml.hw2;

import java.util.Collections;

import org.ejml.data.DenseMatrix64F;

public class Q1aLRL2Spam {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names");
		
//		normalizeData(dataSet);
		Normalize.normalizeDataSandS(dataSet);
		double lambda = 0.1;
		
		errorWithKFold(10, dataSet, lambda);
		
	}
	
private static void errorWithKFold(int k, DataSet dataSet, double lambda) throws Exception {
		
		double avgPerError = 0;
		double avgTrainError = 0;
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
			
			int trainDataSize = dataSet.getData().size();
			int trainFeatureSize = dataSet.getFeatures().size();
			double[] trainFeatureMatrix = dataSet.getFeatureMatrix();
			double[] trainLabelMatrix = dataSet.getLabelMatrix();
			
			DenseMatrix64F weightMatrix = LRImplementation.train(
					trainDataSize,
					trainFeatureSize,
					trainFeatureMatrix,
					trainLabelMatrix,
					lambda);
	
			double trainingError = LRImplementation.calculateError(
					weightMatrix,
					trainDataSize,
					trainFeatureSize,
					trainFeatureMatrix,
					trainLabelMatrix);
			
			avgTrainError += trainingError;
			
			int testDataSize = testData.getData().size();
			int testFeatureSize = testData.getFeatures().size();
			double[] testFeatureMatrix = testData.getFeatureMatrix();
			double[] testLabelMatrix = testData.getLabelMatrix();

			double testingError = LRImplementation.calculateError(weightMatrix,
					testDataSize,
					testFeatureSize,
					testFeatureMatrix,
					testLabelMatrix);
			
			avgPerError += testingError;
		
			System.out.println("Fold " + (fold+1) + " -> Testing Error : " + (1- (avgPerError/ k)));
			System.out.println("Fold " + (fold+1) + " -> Training Error : " + (1 - (avgTrainError/ k)));
			
		}
		
		System.out.println( (lambda) + " -> Testing Error : " + (1- (avgPerError/ k)));
		System.out.println( (lambda) + " -> Training Error : " + (1 - (avgTrainError/ k)));
		
	}

}

package com.ml.hw2;

import org.ejml.simple.SimpleMatrix;

public class GradDesImplement {
	
	private static double lambda;
	private static SimpleMatrix ww;

	public static void learn(DataSet dataSet, double ilambda, boolean logistic,
			double threshold, boolean classification, DataSet testingData) throws Exception {
		
		lambda = ilambda;
		int trainDataSize = dataSet.getData().size();
		int trainFeatureSize = dataSet.getFeatures().size();
		double[] trainFeatureMatrix = dataSet.getFeatureMatrix();
		double[] trainLabelMatrix = dataSet.getLabelMatrix();
		
		double[] weights = new double[trainFeatureSize];
		SimpleMatrix x = new SimpleMatrix(trainDataSize, trainFeatureSize, true, trainFeatureMatrix);
		SimpleMatrix y = new SimpleMatrix(trainDataSize, 1, true, trainLabelMatrix);
		
		for(int i = 0; i< trainFeatureSize; i++){
			weights[i] = Math.random();
		}
		
		SimpleMatrix w = new SimpleMatrix(weights.length, 1, true, weights);
		
		double origError = findErrorMSE(w, x, y, classification);
		int iteration = 0;
		while(iteration < 4000){
			double error = 0;
			for(int row = 0; row < x.numRows(); row++) {
				double predictedValue = 0;
				for(int col=0; col < x.numCols(); col++) {
					predictedValue+= w.get(col,0) * x.get(row, col);
					
				}
				if(logistic) {
					predictedValue = logisticValues(predictedValue);
				}
				for(int col=0; col < x.numCols(); col++) {
					double newWeightValue = updateWeight(w.get(col,0), predictedValue, y.get(row, 0), x.get(row, col));
					w.set(col, 0, newWeightValue);
				}

//				error = findErrorMSE(w, x, y, classification);
//				if(Math.abs(origError-error) < threshold){
//					return;
//				}
				origError = error;
				
			}
			iteration++;
			ww = w;
		}
//		if(!logistic){
			System.out.println("Training Error : " + findErrorMSE(w, x, y, classification));
//		} else {
//			System.out.println("Training Error : " + (100 - findErrorMSE(w, x, y, classification)));
//		}
		
		test(testingData, w, logistic, classification);
		
	}

	private static void test(DataSet testingData, SimpleMatrix w, boolean logistic, boolean classification) throws Exception {
		
		int trainDataSize = testingData.getData().size();
		int trainFeatureSize = testingData.getFeatures().size();
		double[] trainFeatureMatrix = testingData.getFeatureMatrix();
		double[] trainLabelMatrix = testingData.getLabelMatrix();
		
		SimpleMatrix x = new SimpleMatrix(trainDataSize, trainFeatureSize, true, trainFeatureMatrix);
		SimpleMatrix y = new SimpleMatrix(trainDataSize, 1, true, trainLabelMatrix);
		
//		if(!logistic){
			System.out.println("Testing Error : " + findErrorMSE(w, x, y, classification));
//		} else {
//			System.out.println("Testing Error : " + (100 - findErrorMSE(w, x, y, classification)));
//		}
	}

	private static double updateWeight(double old, double predictedValue, double y, double x) {
		double newW = old - lambda * (predictedValue - y) * x;
		return newW;
	}

	private static double logisticValues(double predictedValue) {
		return 1.0/(1.0 + Math.exp(-predictedValue));
	}

	private static double findErrorMSE(SimpleMatrix w, SimpleMatrix x, SimpleMatrix y, boolean classification) {

		SimpleMatrix predY = x.mult(w);
		SimpleMatrix predictedClass = new SimpleMatrix(predY.numRows(), 1);
		
		if(classification){
			for(int row=0; row < predY.numRows(); row++) {
				for(int col=0; col < predY.numCols(); col++) {
					predictedClass.set(row, col, predY.get(row, col) > 0.5 ? 1 : 0);
				}
			}
			double error = 0;
			for(int row=0; row < predY.numRows(); row++) {
				for(int col=0; col < predY.numCols(); col++) {
					error+= (y.get(row, col) != predictedClass.get(row, col)) ? 1 : 0;
				}
			}
			double errorPercentage = error/y.numRows()*100;
			return 100 - errorPercentage;
			
		} else {
			
			SimpleMatrix err = predY.minus(y);
			int rows = err.numRows();
			double sqErr = 0;
			for(int i=0; i<rows;i++) {
				sqErr += Math.pow(err.get(i, 0),2);
			}
			double returnValue = sqErr/rows;
			return returnValue;
			
		}
	}

	public static SimpleMatrix getWeights(DataSet trainingData, double d,
			boolean b, int i, boolean c, DataSet testData) throws Exception {
		
		learn(trainingData, 0.0001, true, 0, true, testData);
		
		
		return ww;
	}

	
}

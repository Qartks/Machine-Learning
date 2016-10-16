package com.ml.hw5;

import org.ejml.simple.SimpleMatrix;

public class GradDesImplementL2 {
	
	private static double lambda;
	private static double regFac;
	private static SimpleMatrix ww;

	public static void learn(DataSet dataSet, double ilambda, boolean logistic,
			double threshold, boolean classification, DataSet testingData, double iregFac) throws Exception {
		
		lambda = ilambda;
		regFac = iregFac;
		int trainDataSize = dataSet.getData().size();
		int trainFeatureSize = dataSet.getFeatureSize();
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
			double[] roundWeightUpdate = new double[dataSet.getFeatureSize()];
			
			for(int row = 0; row < x.numRows(); row++) {
				double predictedValue = 0;
				for(int col=0; col < x.numCols(); col++) {
					predictedValue+= w.get(col,0) * x.get(row, col);
				}
				if(logistic) {
					predictedValue = logisticValues(predictedValue);
				}
				double error = predictedValue - y.get(row, 0);
				for(int col=0; col < x.numCols(); col++) {
					roundWeightUpdate[col]+= error * x.get(row, col);
				}
			}
			for(int col=0; col < x.numCols(); col++) {
				double newWeightValue = updateWeight(w.get(col,0), roundWeightUpdate, col, x.numRows());
				w.set(col, 0, newWeightValue);
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

	private static double updateWeight(double wOld, double[] roundWeightUpdate, int col, int numRows) {
		double value = roundWeightUpdate[col];
		if(col == 0){
			value = value * lambda/numRows;
			return wOld - value;
		}
		value/= numRows;
		value+= wOld * regFac / numRows;
		value*= lambda;
		return wOld - value;
	}

	private static void test(DataSet testingData, SimpleMatrix w, boolean logistic, boolean classification) throws Exception {
		
		int trainDataSize = testingData.getData().size();
		int trainFeatureSize = testingData.getFeatureSize();
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
	
}

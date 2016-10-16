package com.ml.hw2;

import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class GDImplementation {
	
	
	public static void learn(DataSet dataSet, double lambda, boolean regression, double threshold) throws Exception{
		
		int trainDataSize = dataSet.getData().size();
		int trainFeatureSize = dataSet.getFeatures().size();
		double[] trainFeatureMatrix = dataSet.getFeatureMatrix();
		double[] trainLabelMatrix = dataSet.getLabelMatrix();
		
		double[] weights = new double[trainFeatureSize];
		SimpleMatrix x = new SimpleMatrix(trainDataSize, trainFeatureSize, true, trainFeatureMatrix);
		SimpleMatrix y = new SimpleMatrix(trainDataSize, 1, true, trainLabelMatrix);
		
		double weightChange = 0;
		double newError = 0;
		for(int i = 0; i< trainFeatureSize; i++){
			weights[i] = 0;
		}
		
		SimpleMatrix w = new SimpleMatrix(weights.length, 1, true, weights);
		
		double origError = findErrorMSE(w, x, y, regression);
		
		int iter = 1;
		do {
			for(int i = 0; i< trainDataSize; i++){
				
				SimpleMatrix row = x.extractVector(true, i);
				
//				double yj = y.get(i, 0);
//				double[] rowData = row.extractVector(true, 0).getMatrix().getData();
//				
//				double hx = calculateH(x, w, i, regression);
				
				for(int j = 0; j< weights.length; j++){
					double xj = row.get(j);
					updateRule(w, x, y, i, j, lambda);
				}
				
			}
			
			System.out.println("Iteration "+ (iter++) + ".");
			System.out.println( Arrays.toString(weights));
			
			newError = findErrorMSE(w, x, y, regression);
			weightChange = Math.abs(origError - newError);
			origError = newError;
			
		} while( weightChange >  threshold);
		
		System.out.println( Arrays.toString(weights));
		
	}
	
	

	private static void updateRule(SimpleMatrix w, SimpleMatrix x,
			SimpleMatrix y, int i, int j, double lambda) {
		double wOld = w.get(j, 0);
		SimpleMatrix row = x.extractVector(true, i);
		double xj = row.get(j);
		SimpleMatrix pV = row.mult(w);
		double hx = pV.getMatrix().getData()[0];
		double yj = y.get(j, 0);
		double wNew = wOld - lambda * (hx - yj) * xj;
		w.set(j, 0, wNew);;
	}



	private static double findErrorMSE(SimpleMatrix w, SimpleMatrix x, SimpleMatrix y, boolean regression) {
		double avgError = 0;
		SimpleMatrix err = x.mult(w).minus(y);
		if(regression){
			//for(int i = 0; i< x.numRows(); i++){
				for(int k = 0 ; k < err.numRows(); k++){
					avgError += Math.pow(err.get(k, 0), 2);
				}
			//}
			System.out.println(avgError/x.numRows());
			return avgError/x.numRows();
		} else {
			return 0;
		}
	}

//	private static SimpleMatrix updateRule(double[] weights, int j, double hx,double lambda, double xj, double yj) {
//		
//		weights[j] = weights[j] - (lambda * ( hx - yj ) * xj) ;
//		
//		return new SimpleMatrix(weights.length, 1, true, weights);
//	}

	private static double calculateH(SimpleMatrix x, SimpleMatrix w, int i, boolean regression) {
		
		double hx = 0;
		
		SimpleMatrix row = x.extractVector(true, i);
		SimpleMatrix pV = row.mult(w);
		hx = pV.getMatrix().getData()[0];
		
		if(regression){
			hx = 1.0/(1 + Math.exp(-hx));
		} 
		
		return hx;
	}

}

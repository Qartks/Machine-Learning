package com.ml.hw3;

import org.apache.commons.math3.stat.correlation.Covariance;
import org.ejml.simple.SimpleMatrix;


public class GDAImplementation {
	
	static int noOfData = 4601;
	static int noOFFeatures = 57;
	
	public GDAModel train (DataSet dataSet){
		GDAModel model = new GDAModel(noOFFeatures, noOfData);
		double[] zeroµ;
		double[] oneµ;
		double[][] featArr = dataSet.getFeatureArray();
		Covariance cov = new Covariance(featArr);
		model.setCovarianceMatrix(cov.getCovarianceMatrix().getData());
		
//		SimpleMatrix m = new SimpleMatrix(cov.getCovarianceMatrix().getData());
		
		zeroµ = getMeanVector(dataSet, 0);
		oneµ = getMeanVector(dataSet, 1);
		
		model.setZeroµ(zeroµ);
		model.setOneµ(oneµ);
		
		return model;
	}
	
	
	private double[] getMeanVector(DataSet dataSet, double x) {
		double[] mean = new double[dataSet.getFeatureSize()];
		double countOfX = 0;
		for(int i = 0; i< dataSet.getDataSize(); i++){
			Data d = dataSet.getData().get(i);
			if(d.getLabelValue() == x){
				for(int j = 0; j< dataSet.getFeatureSize(); j++){
					mean[j] += d.getFeatureValueAtIndex(j);
				}
				countOfX++;
			}
		}
		for(int j = 0; j< dataSet.getFeatureSize(); j++){
			mean[j] /= countOfX ;
		}
		return mean;
	}


	public double test(GDAModel model, DataSet testData){
		double error = 0;
		for(int i = 0; i< testData.getDataSize(); i++){
			Data d = testData.getData().get(i);
			d.setDataSet(testData);
			double actualValue = d.getLabelValue();
			double predictedValue = computeLabel(model, d, testData.getDataSize());
			if(predictedValue != actualValue){
				error++;
			}
		}
		double accuracy = 100d - (error/testData.getDataSize() * 100);
		return accuracy;
	}


	private double computeLabel(GDAModel model, Data d, int size) {
		double probOfOne = d.getDataSet().getProbabilityOfOne();
		double probOfZero = 1 - probOfOne;
		
		SimpleMatrix x = new SimpleMatrix(1, d.getFeatureValues().length, true, d.getFeatureValues());
		SimpleMatrix µ1 = new SimpleMatrix(1, model.getOneµ().length, true, model.getOneµ());
		SimpleMatrix µ0 = new SimpleMatrix(1, model.getZeroµ().length, true, model.getZeroµ());
		SimpleMatrix cov = new SimpleMatrix(model.getCovarianceMatrix());
		
		double probX_Y_0 = calculateProbabilityWithLog(probOfZero,x, µ0, cov, size);
		double probX_Y_1 = calculateProbabilityWithLog(probOfOne,x, µ1, cov, size);
		
		if(probX_Y_0 >= probX_Y_1){
			return 0;
		} else {
			return 1;
		}
	}


	private double calculateProbabilityWithLog(double probY, SimpleMatrix x, SimpleMatrix µ, SimpleMatrix cov, int dataSize) {
	
		SimpleMatrix xMinusµ = x.minus(µ).transpose();
		SimpleMatrix xMinusµTrans = xMinusµ.transpose();
		SimpleMatrix exp = xMinusµTrans.mult(cov.invert()).mult(xMinusµ);
		double nLog2Pi = dataSize * Math.log(Math.PI*2);
		double logCov = Math.log(cov.determinant());
		double expVal = exp.determinant();
		double logPY = Math.log(probY);
		return logPY - (nLog2Pi + logCov + expVal) * 0.5d;
	}


	private double calculateProbability(double probY, SimpleMatrix x, SimpleMatrix µ, SimpleMatrix cov, int dataSize) {
		SimpleMatrix xMinusµ = x.minus(µ).transpose();
		SimpleMatrix xMinusµTrans = xMinusµ.transpose();
		SimpleMatrix exp = xMinusµTrans.mult(cov.invert()).mult(xMinusµ);
//		double nLog2Pi = dataSize/2 * Math.
		double denom = (Math.pow(Math.PI*2, dataSize/2)) * Math.sqrt((cov.determinant()));
		double expVal = Math.exp(exp.determinant() * -0.5);
		return (1/denom) * expVal * probY;
	}
	

}

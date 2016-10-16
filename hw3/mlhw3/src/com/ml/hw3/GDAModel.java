package com.ml.hw3;

public class GDAModel {
	
	double[][] covarianceMatrix;
	double[] zeroµ;
	double[] oneµ;
	
	public GDAModel(int featureSize, int dataSize){
		covarianceMatrix = new double[featureSize][featureSize];
		zeroµ = new double[featureSize];
		oneµ = new double[featureSize];
	}

	public double[][] getCovarianceMatrix() {
		return covarianceMatrix;
	}

	public void setCovarianceMatrix(double[][] covarianceMatrix) {
		this.covarianceMatrix = covarianceMatrix;
	}

	public double[] getZeroµ() {
		return zeroµ;
	}

	public void setZeroµ(double[] zeroµ) {
		this.zeroµ = zeroµ;
	}

	public double[] getOneµ() {
		return oneµ;
	}

	public void setOneµ(double[] oneµ) {
		this.oneµ = oneµ;
	}
	
	

}

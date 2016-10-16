package com.ml.hw4;

import java.util.Arrays;

public class FeatureStats {
	
	int noOfFeatures;
	double[][] featureStats;
	
	public FeatureStats(int n) {
		noOfFeatures = n;
		featureStats = new double[noOfFeatures][7];
	}
	
	public double[][] getFeatureStats() {
		return featureStats;
	}

	public void setFeatureStats(double[][] featureStats) {
		this.featureStats = featureStats;
	}

	public double getMinOfFeature(int i){
		return featureStats[i][0];
	}

	public double getMaxOfFeature(int i){
		return featureStats[i][1];
	}
	
	public double getMeanOfFeature(int i){
		return featureStats[i][2];
	}
	
	public double getSdOfFeature(int i){
		return featureStats[i][3];
	}

	public double getCoeffVarPerOfFeature(int i){
		return featureStats[i][4];
	}
	
	public double getMeanSpam(int i){
		return featureStats[i][5];
	}
	
	public double getMeanNoSpam(int i){
		return featureStats[i][6];
	}
	
	public void setValue(int fNo, int index, double val){
		featureStats[fNo][index] = val;
	}

	public double[][] getSortedFeatureStats() {
		double[][]fs = new double[57][5];
		
		for(int i = 0; i< 57; i++){
			fs[i][0] = featureStats[i][0];
			fs[i][1] = featureStats[i][1];
			fs[i][2] = featureStats[i][2];
			fs[i][3] = featureStats[i][5];
			fs[i][4] = featureStats[i][6];
			
			Arrays.sort(fs[i]);
		}
		return fs;
	}
	

}

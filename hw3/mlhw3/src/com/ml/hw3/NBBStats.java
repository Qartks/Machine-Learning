package com.ml.hw3;

public class NBBStats {

	double[][] probStats;
	boolean spam;

	public NBBStats(int n) {
		probStats = new double[n][4];
		spam = false;
	}

	public double[][] getProbStats() {
		return probStats;
	}

	public void setProbStats(double[][] probStats) {
		this.probStats = probStats;
	}
	
	public void setSpam(boolean b){
		this.spam = b;
	}

	public double getlessMean(int i){
		if(spam){
			return probStats[i][0];
		} else {
			return probStats[i][2];
		}
	}

	public double getGreatMean(int i){
		if(spam){
			return probStats[i][1];
		} else {
			return probStats[i][3];
		}
	}
	
	public void setlessMeanSpam(int i,double val){
		probStats[i][0] = val;
	}

	public void setGreatMeanSpam(int i,double val){
		probStats[i][1] = val;
	}
	
	public void setlessMeanNotSpam(int i,double val){
		probStats[i][2] = val;
	}

	public void setGreatMeanNotSpam(int i,double val){
		probStats[i][3] = val;
	}
}

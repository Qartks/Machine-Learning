package com.ml.hw3;

public class Bucket4 {
	
	double[][] spamProb;
	double[][] noSpamProb;
	double[][] featureBuckets;
	boolean spam;
	
	public Bucket4() {
		spamProb = new double[57][4];
		noSpamProb = new double[57][4];
		spam = false;
		featureBuckets = new double[57][5];
	}
	
	public double[][] getSpamProb() {
		return spamProb;
	}

	public void setSpamProb(double[][] spamProb) {
		this.spamProb = spamProb;
	}

	public double[][] getNoSpamProb() {
		return noSpamProb;
	}

	public void setNoSpamProb(double[][] noSpamProb) {
		this.noSpamProb = noSpamProb;
	}

	public boolean isSpam() {
		return spam;
	}

	public void setSpam(boolean spam) {
		this.spam = spam;
	}

	public void setBuckets(double[][] featureStats) {
		this.featureBuckets = featureStats;
	}
	
	public double[][] getBuckets() {
		return this.featureBuckets;
	}

	public double[][] getProbs() {
		if(spam){
			return getSpamProb();
		} else {
			return getNoSpamProb();
		}
	}

}

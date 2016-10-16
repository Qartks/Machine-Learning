package com.ml.hw3;

public class Bucket9 {
	
	double[][] spamProb;
	double[][] noSpamProb;
	FeatureStats feat;
	boolean spam;
	
	public Bucket9() {
		spamProb = new double[57][9];
		noSpamProb = new double[57][9];
		feat = null;
	}
	
	public boolean isSpam() {
		return spam;
	}

	public void setSpam(boolean spam) {
		this.spam = spam;
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

	public void setTrainingStats(FeatureStats fs) {
		this.feat = fs;
	}
	
	public FeatureStats getTrainingStats() {
		return this.feat;
	}

	public double[][] getProbs() {
		if(spam){
			return getSpamProb();
		} else {
			return getNoSpamProb();
		}
	}
	
}

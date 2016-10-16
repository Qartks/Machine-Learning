package com.ml.hw5;

public class GaussParams {
	
	FeatureStats spam;
	FeatureStats noSpam;
	FeatureStats allData;
	double ProbOfSpam;
	
	public FeatureStats getSpam() {
		return spam;
	}
	public void setSpam(FeatureStats spam) {
		this.spam = spam;
	}
	public FeatureStats getNoSpam() {
		return noSpam;
	}
	public void setNoSpam(FeatureStats noSpam) {
		this.noSpam = noSpam;
	}
	public FeatureStats getAllData() {
		return allData;
	}
	public void setAllData(FeatureStats all) {
		this.allData = all;
	}

}

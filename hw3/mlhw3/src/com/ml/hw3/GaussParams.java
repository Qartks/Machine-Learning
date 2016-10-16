package com.ml.hw3;

public class GaussParams {
	
	FeatureStats spam;
	FeatureStats noSpam;
	FeatureStats allData;
	
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
	public void setAllData(FeatureStats noSpam) {
		this.allData = noSpam;
	}

}

package com.ml.hw5;

public class Feature {
	
	public static final int NUMERICAL = 1;
	public static final int NOMINAL = 2;
	
	private final String featureName;
	private final int featureType;
	
	public Feature(String featureName, int featureType) {
		this.featureName = featureName;
		this.featureType = featureType;
	}
	
	public boolean isNumerical() {
		return this.featureType == NUMERICAL;
	}
	
	public boolean isNominal() {
		return this.featureType == NOMINAL;
	}
	
	public String getFeatureName() {
		return this.featureName;
	}

}

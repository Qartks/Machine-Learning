package com.ml.hw7;

public class FeatureValue {
	int featureId;
	double distance;
	
	public FeatureValue(int f, double d) {
		this.featureId = f;
		this.distance = d;
	}

	@Override
	public String toString() {
		return "FeatureValue [featureId=" + featureId + ", distance=" + distance + "]";
	}
	
}

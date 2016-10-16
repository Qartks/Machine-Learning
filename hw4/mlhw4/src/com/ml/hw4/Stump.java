package com.ml.hw4;

public class Stump {

	int featureIndex;
	double thresholdValue;
	DataSet dataSet;
	double[] weights;
	
	double error;
	
	public Stump(DataSet td, double[] d){
		error = Double.NEGATIVE_INFINITY;
		weights = d;
		dataSet = td;
	}
	
	public void calculateError(){
		error = Math.abs(0.5 - this.getEpsilon());
	}
	
	public double getError(){
		return error;
	}
	
	public int getFeatureIndex() {
		return featureIndex;
	}

	public void setFeatureIndex(int featureIndex) {
		this.featureIndex = featureIndex;
	}

	public double getThresholdValue() {
		return thresholdValue;
	}

	public void setThresholdValue(double thresholdValue) {
		this.thresholdValue = thresholdValue;
	}
	
	public double getEpsilon() {
		double sum = 0;
		for(int i = 0; i<this.dataSet.getDataSize(); i++){
			Data data = dataSet.getData().get(i);
			double predictedValue = this.getPredicted(data);
			double actualValue = data.labelValue == 0 ? -1 : 1;
			
			if(predictedValue != actualValue){
				sum += weights[i];
			}
		}
		
		return sum;
	}

	public double getPredicted(Data data) {
		double fVal = data.getFeatureValueAtIndex(featureIndex);
		if(fVal > thresholdValue){
			return 1;
		}
		return -1;
	}
}

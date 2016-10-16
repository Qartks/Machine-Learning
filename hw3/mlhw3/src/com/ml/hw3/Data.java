package com.ml.hw3;

public class Data {
	
	double[] featureValues;
	double labelValue;
	private DataSet dataSet;
	
	public Data(int size){
		featureValues = new double[size]; 
	}

	public double[] getFeatureValues() {
		return featureValues;
	}

	public void setFeatureValues(double[] featureValues) {
		this.featureValues = featureValues;
	}
	
	public double  getFeatureValueAtIndex(int i) {
		return featureValues[i];
	}

	public void setFeatureValueAtIndex(int i, double val) {
		featureValues[i] = val;
	}

	public double getLabelValue() {
		return labelValue;
	}

	public void setLabelValue(double labelValue) {
		this.labelValue = labelValue;
	}

	public void setDataSet(DataSet dataSet) {
		this.dataSet = dataSet;
	}
	
	public DataSet getDataSet() {
		return this.dataSet;
	}

	
}

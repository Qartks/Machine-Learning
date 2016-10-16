package com.ml.hw4;

import java.util.List;

public class Data2 {
	
	private double[] featureValues;
	private DataSet2 dataset;
	private double originalLabel;

	public Data2(Data2 data) {
		featureValues = data.getFeatureValues();
		dataset = data.getDataSet();
	}

	public Data2(double[] values) {
		this.featureValues = values;
		this.dataset = null;
	}
	
	public double[] getFeatureValues() {
		return featureValues;
	}
	

	public double getFeatureValue(int index) {
		return featureValues[index];
	}
	

	public DataSet2 getDataSet() {
		return dataset;
	}
	

	public void setDataSet(DataSet2 dataset) {
		this.dataset = dataset;
	}
	

	public int labelIndex() throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getLabelIndex();
	}
	

	public double labelValue() throws Exception {
		return featureValues[labelIndex()];
	}
	
	public void setLabelValue(double val) throws Exception {
		featureValues[labelIndex()] = val;
	}

	public double getOriginalLabel() {
		return originalLabel;
	}

	public void setOriginalLabel(double originalLabel) {
		this.originalLabel = originalLabel;
	}
	
	public Feature getFeature(int index) throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getFeature(index);
	}

	public List<Feature> getFeatures() throws Exception {
		if(dataset == null) {
			throw new Exception("Dataset is null");
		}
		return dataset.getFeatures();
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new  StringBuilder();
		for(double value : featureValues) {
			builder.append(String.valueOf(value)+" ");
		}
		return builder.toString().trim();
	}
}

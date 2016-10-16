package com.ml.hw6;

import java.util.List;

public class Data {
	
	double[] featureValues;
	double labelValue;
	private DataSet dataSet;
	public List<Integer> missingValues;
	
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

	public void calculateDataMissingValues(FeatureStats featStat) {
		for(int i : missingValues){
			int maxFNo = 0;
			double maxMean = Double.NEGATIVE_INFINITY;
			if(this.dataSet.featureProp.containsKey(i)){
				List<String> list = this.dataSet.featureProp.get(i);
				int fNo = this.dataSet.featureMap.get(i);
				for(int j = 0; j < list.size(); j++){
					if(featStat.getMeanOfFeature(j+fNo) > maxMean){
						maxMean = featStat.getMeanOfFeature(j+fNo);
						maxFNo = j+fNo;
					}
				}
				featureValues[maxFNo] = 1;
			} else {
				int fNo = this.dataSet.featureMap.get(i);
				featureValues[fNo] = featStat.getMeanOfFeature(fNo);
			}
		}
	}

	public String toString(){
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i< featureValues.length; i++){
			sb.append(featureValues[i]);
			sb.append(" ");
		}
		sb.append((double)this.getLabelValue());
		
		return sb.toString().trim();
	}
	
}

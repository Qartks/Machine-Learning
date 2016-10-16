package com.ml.hw5;

import java.util.ArrayList;
import java.util.List;

public class RankFeature {
	
	int featureIndex;
	double featureThreshold;
	int count;
	List<Integer> indices;
	double[] π;
	double margin;
	
	public RankFeature() {
		
	}
	
	public RankFeature(int f, int c) {
		this.featureIndex = f;
		this.count = c;
	}
	
	public RankFeature(int f, double thres, double[] iπ) {
		this.featureIndex = f;
		this.featureThreshold = thres;
		this.π = iπ;
		this.indices = new ArrayList<Integer>();
	}
	
	public int getFeatureIndex() {
		return featureIndex;
	}
	public void setFeatureIndex(int featureIndex) {
		this.featureIndex = featureIndex;
	}
	public double getFeatureThreshold() {
		return featureThreshold;
	}
	public void setFeatureThreshold(double featureThreshold) {
		this.featureThreshold = featureThreshold;
	}
	public int getCount() {
		return count;
	}
	public void setCount(int count) {
		this.count = count;
	}
	public List<Integer> getIndices() {
		return indices;
	}
	public void setIndices(List<Integer> indices) {
		this.indices = indices;
	}
	
	public void calculateMargin(List<DTreeStump> models, DataSet dataSet, double fullMargin){
		double avgMar = 0;
		for(int i = 0; i< dataSet.getDataSize(); i++){
			double m = 0;
			double sum = 0;
			Data d = dataSet.getData().get(i);
			for(int ind : indices){
				DTreeStump model = models.get(ind);
				m += (model.getPredicted(d) * π[ind]);
//				sum += Math.abs(π[ind]);
			}
			double yi = d.getLabelValue() == 0 ? -1 : 1;
			double avg = m * yi; 
//			avgMar += (avg/sum);
			avgMar += avg;
		}
//		avgMar /= dataSet.getDataSize();
		
		this.margin = avgMar/fullMargin;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + featureIndex;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		RankFeature other = (RankFeature) obj;
		if (featureIndex != other.featureIndex)
			return false;
		return true;
	}

	@Override
	public String toString() {
		return "RankFeature [featureIndex=" + featureIndex + ", margin="
				+ margin + "]";
	}

	public void incrementCount(int i) {
		this.count++;
		this.indices.add(i);
	}
	
	
}

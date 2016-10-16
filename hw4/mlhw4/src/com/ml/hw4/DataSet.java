package com.ml.hw4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;


public class DataSet {
	
	private List<Data> data;
	private int featureSize;
	private FeatureStats featStat;
	public HashMap<Integer, List<String>> featureProp;
	public HashMap<Integer, Integer> featureMap;

	public DataSet(int noOFFeatures){
		data = new ArrayList<Data>();
		featureSize = noOFFeatures;
	}
	
	public List<Data> getData() {
		return data;
	}
	
	public int getDataSize(){
		return data.size();
	}

	public void setData(List<Data> data) {
		this.data = data;
	}

	public void addData(Data d){
		data.add(d);
	}
	
	public void removeData(Data d){
		data.remove(d);
	}
	
	public double getProbabilityOfOne(){
		double count = 0;
		
		for(Data d: this.getData()){
			if(d.getLabelValue() == 1){
				count++;
			}
		}
		
		return (count + 1)/(this.getDataSize() + 2);
	}

	public double[][] getFeatureArray() {
		double[][] featureArray = new double[this.getDataSize()][featureSize];
		for(int i = 0; i< this.getDataSize(); i++){
			Data d = data.get(i);
			for(int j = 0; j < featureSize; j++){
				featureArray[i][j] = d.getFeatureValueAtIndex(j);
			}
		}
		return featureArray;
	}

	public void setFeatureSize(int noOFFeatures) {
		this.featureSize = noOFFeatures;
	}
	
	public int getFeatureSize(){
		return featureSize;
	}
	
	public void setFeatureStat(FeatureStats fs){
		this.featStat = fs;
	}
	
	public FeatureStats getFeatureStat(){
		return this.featStat;
	}

	public void computeFeatureStats() {
		this.featStat = new FeatureStats(featureSize);
		
		double[] mean = new double[featureSize];
		double[] var = new double[featureSize];
		double[] min = new double[featureSize];
		double[] max = new double[featureSize];
		
		Arrays.fill(min, Double.MAX_VALUE);
		Arrays.fill(max, Double.MIN_VALUE);
		
		for(Data d : this.getData()){
			for(int i = 0; i< featureSize; i++){
				double val = d.getFeatureValueAtIndex(i);
				mean[i] += val;
				if(val > max[i]){
					max[i] = val;
				}
				if(val < min[i]){
					min[i] = val;
				}
			}
		}
		
		for(int i = 0; i< featureSize; i++){
			mean[i] /= this.data.size();
		}
		
		for(Data d : this.getData()){
			for(int i = 0; i< featureSize; i++){
				var[i] += Math.pow((d.getFeatureValueAtIndex(i) - mean[i]),2);
			}
		}
		
		for(int i = 0; i< featureSize; i++){
			var[i] /= this.data.size();
		}
		
		
		
		for(int i = 0; i< featureSize; i++){
			featStat.setValue(i, 2, mean[i]);
			featStat.setValue(i, 4, var[i]);
			featStat.setValue(i, 0, min[i]);
			featStat.setValue(i, 1, max[i]);
		}
	}

	public void calculateMissingValues() {
		for(Data d : this.getData()){
			d.calculateDataMissingValues(this.featStat);
		}
		
	}
	
}

package com.ml.hw7;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;


public class DataSet {
	
	private List<Data> data;
	private int featureSize;
	private FeatureStats featStat;
	public HashMap<Integer, List<String>> featureProp;
	public HashMap<Integer, Integer> featureMap;
	Map<Integer, Set<Double>> map;

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
		
		return count/(this.getDataSize());
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
				if(Double.isNaN(val)){
					continue;
				}
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
				double val = d.getFeatureValueAtIndex(i);
				if(Double.isNaN(val)){
					continue;
				}
				var[i] += Math.pow((val - mean[i]),2);
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

	public double[] getFeatureMatrix() {
		double featureSize = this.getFeatureSize();
		double dataSize = this.getData().size();
		
		double[] featureMatrix = new double[(int) (featureSize * dataSize)];
		int i = 0;
		
		for(Data d : this.getData()){
			featureMatrix[i] = 1;
			i++;
			for(int j = 0; j <featureSize - 1; j++){
				featureMatrix[i] = d.getFeatureValueAtIndex(j);
				i++;
			}
		}
		
		return featureMatrix;
	}

	public double[] getLabelMatrix() {
		double[] labelMatrix = new double[this.getDataSize()];
		
		for(int i = 0; i< this.getData().size(); i++){
			labelMatrix[i] = this.getData().get(i).getLabelValue();
		}
		
		return labelMatrix;
	}

	public void computeOptimalThreVal() {
		this.map = getThresListMap();
	}
	
	private Map<Integer, Set<Double>> getThresListMap() {
		Map<Integer, Set<Double>> map = new HashMap<Integer, Set<Double>>();
		for(int f = 0; f < this.getFeatureSize(); f++){
			Set<Double> list = new TreeSet<Double>();
			double threshold = 0;
			for(int i = 0; i < this.getDataSize() - 1; i++){
				Data data1 = this.getData().get(i);
				Data data2 = this.getData().get(i+1);
				
				if(data1.getFeatureValueAtIndex(f) != data2.getFeatureValueAtIndex(f)){
					threshold = (data1.getFeatureValueAtIndex(f) + data2.getFeatureValueAtIndex(f))/2;
					list.add(threshold);
				}
			}
			list.add(this.getFeatureStat().getMaxOfFeature(f) + 1);
			list.add(this.getFeatureStat().getMinOfFeature(f) - 1);
			map.put(f, list);
		}
		return map;
	}
	
}

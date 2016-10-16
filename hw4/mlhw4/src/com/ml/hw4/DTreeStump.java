package com.ml.hw4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;


public class DTreeStump implements Cloneable {
	
	DataSet dataSet;
	double[] weights;
	boolean optimal;
	
	int featureIndex;
	double thresholdValue;
	
	HashMap<Integer, List<Double>> featureHist;
	
	protected Object clone() throws CloneNotSupportedException {
        return super.clone();
    }
	
	public DTreeStump(DataSet trainData) {
		this.dataSet = trainData;
		featureHist = new HashMap<Integer, List<Double>>();
	}

	public DTreeStump(DataSet trainData, double[] d, boolean optimal) {
		this.dataSet = trainData;
		this.weights = d;
		this.optimal = optimal;
		featureHist = new HashMap<Integer, List<Double>>();
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
		if(fVal >= thresholdValue){
			return 1;
		}
		return -1;
	}
	
	public void getRandomStump(){
		dataSet.computeFeatureStats();
		FeatureStats fs = dataSet.getFeatureStat();
		
		int f = (int) (Math.random() * dataSet.getFeatureSize());
		double t = fs.getMinOfFeature(f) + (Math.random() * fs.getMaxOfFeature(f) - fs.getMinOfFeature(f));
		
		this.featureIndex = f;
		this.thresholdValue = t;
	}

	public void generateBestStump() {
//		dataSet.computeFeatureStats();
		FeatureStats fs = dataSet.getFeatureStat();
		Stump bestStump = new Stump(dataSet, weights);
		int bestFeature = 0;
		for(int f = 0; f < fs.noOfFeatures; f++){
			
			Stump tempStump = createBestSplitOnFeatureBetter(f);
			tempStump.calculateError();
//			System.out.println(tempStump.featureIndex + " -> " + tempStump.thresholdValue + " -> " + tempStump.getError());
			
			if(tempStump.getError() > bestStump.getError()){
				bestStump = tempStump;
			}
			
		}
		
		this.featureIndex = bestStump.getFeatureIndex();
		this.thresholdValue = bestStump.getThresholdValue();
		addThresholdFeatureToHistory(featureIndex,thresholdValue);
	}
	
	private Stump createBestSplitOnFeature(int f) {
//		sortFeatures(dataSet, f);
		Stump temp = new Stump(dataSet, weights);
		
		double bestThreshold = 0;
		double maxError = Double.NEGATIVE_INFINITY;
		
		for(int i = 0; i < dataSet.getDataSize() - 1; i++){
			Stump temp2 = new Stump(dataSet, weights);
			double threshold = 0;
			double err = 0;
			
			Data data1 = dataSet.getData().get(i);
			Data data2 = dataSet.getData().get(i+1);
			
			if(data1.getFeatureValueAtIndex(f) != data2.getFeatureValueAtIndex(f)){
				
				threshold = (data1.getFeatureValueAtIndex(f) + data2.getFeatureValueAtIndex(f))/2;
				
				if(checkHistory(f, bestThreshold)){
					continue;
				}
				
				temp2.setThresholdValue(threshold);
				temp2.setFeatureIndex(f);
				temp2.calculateError();
				err = temp2.getError();
				
				if(err > maxError){
					bestThreshold = threshold;
					maxError = err;
				}
				
			}
		}
		
		
		temp.setFeatureIndex(f);
		temp.setThresholdValue(bestThreshold);
		return temp;
	}
	
	private Stump createBestSplitOnFeatureBetter(int f) {
		Set<Double> threshList = getThresList(dataSet, f);
		Stump temp = new Stump(dataSet, weights);
		
		double bestThreshold = 0;
		double maxError = Double.NEGATIVE_INFINITY;
		
		for(double threshold : threshList){
			Stump temp2 = new Stump(dataSet, weights);
			double err = 0;
			
			if(checkHistory(f, threshold)){
				continue;
			}
			
			temp2.setThresholdValue(threshold);
			temp2.setFeatureIndex(f);
			temp2.calculateError();
			err = temp2.getError();
			
			if(err > maxError){
				bestThreshold = threshold;
				maxError = err;
			}
		}
		
		temp.setFeatureIndex(f);
		temp.setThresholdValue(bestThreshold);
		return temp;
	}

	private Set<Double> getThresList(DataSet dataSet, int f) {
		
		Set<Double> list = new TreeSet<Double>();
		dataSet.computeFeatureStats();
		double threshold = 0;
		for(int i = 0; i < dataSet.getDataSize() - 1; i++){
			Data data1 = dataSet.getData().get(i);
			Data data2 = dataSet.getData().get(i+1);
			
			if(data1.getFeatureValueAtIndex(f) != data2.getFeatureValueAtIndex(f)){
				threshold = (data1.getFeatureValueAtIndex(f) + data2.getFeatureValueAtIndex(f))/2;
				list.add(threshold);
			}
		}
		list.add(dataSet.getFeatureStat().getMaxOfFeature(f) + 1);
		list.add(dataSet.getFeatureStat().getMinOfFeature(f) - 1);
		return list;
	}

	private void addThresholdFeatureToHistory(int f, double bestThreshold) {
		List<Double> fVals;
		if(featureHist.containsKey(f)){
			fVals = featureHist.get(f);
			fVals.add(bestThreshold);
			featureHist.put(f, fVals);
		} else {
			fVals = new ArrayList<Double>();
			fVals.add(bestThreshold);
			featureHist.put(f, fVals);
		}
	}

	private boolean checkHistory(int f, double threshold) {
		if(featureHist.containsKey(f)){
			List<Double> fVals = featureHist.get(f);
			
			for(Double d : fVals){
				if(d == threshold){
					return true;
				}
			}
		}
		
		return false;
	}

	private void sortFeatures(DataSet dataSet,int featureIndex) {
		if(dataSet != null) {
			Collections.sort(dataSet.getData(), new Comparator<Data>() {

				@Override
				public int compare(Data arg0, Data arg1) {
					Double value1 = new Double(arg0.getFeatureValueAtIndex(featureIndex));
					Double value2 = new Double(arg1.getFeatureValueAtIndex(featureIndex));
					return value1.compareTo(value2);
				}
			});
		}
	}

	public double getMetaPredicted(Data data, double[] d, List<DTreeStump> ldt, double[] π, int t) {
		double val = 0;
		for(int i = 0; i < t; i++){
			val += π[i] * ldt.get(i).getPredicted(data);
		}
		return val;
	}

}

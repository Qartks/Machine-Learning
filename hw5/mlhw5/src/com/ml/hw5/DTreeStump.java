package com.ml.hw5;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class DTreeStump {
	
	int featureIndex;
	double thresholdValue;
	double epsilon;
	DataSet dataSet;
	double[] weights;
	
	HashMap<Integer, List<Double>> featureHist;
	
	public DTreeStump(DataSet dataSet, double[] w) {
		this.dataSet = dataSet;
		this.weights = w;
		this.featureHist = new HashMap<Integer, List<Double>>();
	}
	
	public double getPredicted(Data d) {
		double val = d.getFeatureValueAtIndex(featureIndex);
		if(val >= thresholdValue){
			return 1;
		} else {
			return -1;
		}
	}
	
	public void getRandomStump(){
		FeatureStats fs = dataSet.getFeatureStat();
		
		int f = (int) getRandomValue(0,dataSet.getFeatureSize()-1);
		double t = getRandomValue(fs.getMinOfFeature(f) ,fs.getMaxOfFeature(f));
		
		this.featureIndex = f;
		this.thresholdValue = t;
		this.calculateEpsilon();
	}
	
	private double getRandomValue(double start, double end) {
		Random random = new Random();
		double range = end - start;
		double fraction = range * random.nextDouble();
		return fraction + start;
	}
	
	public void calculateEpsilon() {
		double sum = 0;
		for(int i = 0; i<this.dataSet.getDataSize(); i++){
			Data data = dataSet.getData().get(i);
			double predictedValue = this.getPredicted(data);
			double actualValue = data.labelValue == 0 ? -1 : 1;
			
			if(predictedValue != actualValue){
				sum += weights[i];
			}
		}
		
		this.epsilon = sum;
	}

	public void getOptimalStump(Map<Integer, Set<Double>> map) {
		
		int bestFeatureIndex = -1;
		double bestMaxDistanceFromHalf = Double.NEGATIVE_INFINITY;
		double bestFeatureThreshold = -1;
		double bestError = 0;
		
		for(int featureIndex = 0; featureIndex < dataSet.getFeatureSize(); featureIndex++) {
			Set<Double> thresValues = map.get(featureIndex);
			for(double threshold : thresValues) {
				double error = this.testFeatureThreshold(featureIndex, threshold);
				if(Math.abs(0.5 - error) > bestMaxDistanceFromHalf) {
					bestFeatureIndex = featureIndex;
					bestMaxDistanceFromHalf = Math.abs(0.5 - error);
					bestError = error;
					bestFeatureThreshold = threshold;
				}
			}
		}
		
		this.featureIndex = bestFeatureIndex;
		this.thresholdValue = bestFeatureThreshold;
		this.epsilon = bestError;
	}

	private double testFeatureThreshold( int featureIndex, double threshold) {
		this.featureIndex = featureIndex;
		this.thresholdValue = threshold;
		this.calculateEpsilon();
		return this.epsilon;
	}

	


}

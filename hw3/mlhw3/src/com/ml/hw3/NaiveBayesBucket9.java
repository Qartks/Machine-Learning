package com.ml.hw3;

import java.util.ArrayList;
import java.util.List;

public class NaiveBayesBucket9 {
	
	public List<Double> rocData = new ArrayList<Double> ();
	final private int BUCKETS; 
	final private int FEATURE_SIZE; 
	
	public NaiveBayesBucket9(int n, int i) {
		this.BUCKETS = n;
		this.FEATURE_SIZE = i;
	}

	public Bucket9 train(DataSet dataSet){
		Bucket9 b9 = new Bucket9();
		
		dataSet.computeFeatureStats();
		
		FeatureStats fs = dataSet.getFeatureStat();
		double spamCount = 0;
		double noSpamCount = 0;
		
		double[][] spamBucketProb = new double[FEATURE_SIZE][BUCKETS];
		double[][] noSpamBucketProb = new double[FEATURE_SIZE][BUCKETS];
		
		for(Data d : dataSet.getData()){
			if(d.getLabelValue() == 1){
				spamCount++;
				for(int f = 0; f < dataSet.getFeatureSize(); f++){
					double val = d.getFeatureValueAtIndex(f);
					double min = fs.getMinOfFeature(f);
					double max = fs.getMaxOfFeature(f);
					double[] buckets = computeRangeForBuckets(min, max, BUCKETS);
					
					for(int i = 0; i<BUCKETS-1; i++){
						if(val >= buckets[i] && val< buckets[i+1]){
							spamBucketProb[f][i]++;
							break;
						}
					}
				}
			} else {
				noSpamCount++;
				for(int f = 0; f < dataSet.getFeatureSize(); f++){
					double val = d.getFeatureValueAtIndex(f);
					double min = fs.getMinOfFeature(f);
					double max = fs.getMaxOfFeature(f);
					double[] buckets = computeRangeForBuckets(min, max, BUCKETS);
					
					for(int i = 0; i< BUCKETS-1; i++){
						if(val >= buckets[i] && val< buckets[i+1]){
							noSpamBucketProb[f][i]++;
							break;
						}
					}
				}
			}
		}
		
		for(int f = 0; f < FEATURE_SIZE; f++){
			for(int i = 0; i < BUCKETS; i++){
				spamBucketProb[f][i] = (spamBucketProb[f][i] + 1) / (spamCount + 2);
				noSpamBucketProb[f][i] = (noSpamBucketProb[f][i] + 1) / (noSpamCount + 2);
			}
		}
		
		b9.setTrainingStats(fs);
		b9.setNoSpamProb(noSpamBucketProb);
		b9.setSpamProb(spamBucketProb);
		
		return b9;
	}

	private double[] computeRangeForBuckets(double min, double max, int n) {
		double[] buckets = new double[n+1];
		double size = (max - min) / (n + 1);
		
		buckets[n] = Double.MIN_VALUE;
		for(int i = 1; i < n; i++){
			buckets[i] = min + i * size;
		}
		buckets[n] = Double.MAX_VALUE;
		return buckets;
	}

	public ErrorStat test(Bucket9 b9, DataSet testData, double thres) {
		ErrorStat err = new ErrorStat();
		testData.computeFeatureStats();
		for(int i = 0; i< testData.getDataSize(); i++){
			Data d = testData.getData().get(i);
			d.setDataSet(testData);
			double actualValue = d.getLabelValue();
			double predictedValue = computeLabel(b9, d, thres);
			if(predictedValue != actualValue){
				err.error++;
			}
			err.updateErrorRate(predictedValue, actualValue);
		}
		err.setAccuracy(testData.getDataSize());
		return err;
	}

	private double computeLabel(Bucket9 b9, Data d, double thres) {
		double probOfOne = d.getDataSet().getProbabilityOfOne();
		double probOfZero = 1 - probOfOne;
		double[] x = d.getFeatureValues();
		
		b9.setSpam(false);
		double probX_Y_0 = calculateProbability(probOfZero, x, b9);
		b9.setSpam(true);
		double probX_Y_1 = calculateProbability(probOfOne, x, b9);
		
		double ratio = probX_Y_1 / probX_Y_0;
		rocData.add(ratio);
		
//		if((probX_Y_1 / probX_Y_0) >= thres){
		if(probX_Y_1 > probX_Y_0){
			return 1;
		} else {
			return 0;
		}
	}

	private double calculateProbability(double probOfY, double[] x, Bucket9 b9) {
		double prob = 1d;
		for(int i = 0; i < x.length; i++){
			prob *= calculateValue(i , x[i], b9.getProbs(), b9.getTrainingStats());
		}
		return prob * probOfY;
	}

	private double calculateValue(int f, double val, double[][] probs, FeatureStats ts) {
		
		double min = ts.getMinOfFeature(f);
		double max = ts.getMaxOfFeature(f);
		double[] buckets = computeRangeForBuckets(min, max, BUCKETS);
		
		for(int i = 0; i< BUCKETS - 1; i++){
			if(val >= buckets[i] && val< buckets[i+1]){
				return probs[f][i];
			}
		}
		
		return 0;
	}

}

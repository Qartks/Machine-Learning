package com.ml.hw3;

import java.util.ArrayList;
import java.util.List;

public class NavieBayesBucket4 {
	
	public List<Double> rocData = new ArrayList<Double> ();

	public Bucket4 train(DataSet dataSet) {
		Bucket4 b4 = new Bucket4();
		
		DataSet spamDs = new DataSet(57);
		DataSet noSpamDs = new DataSet(57);
		
		for(Data d : dataSet.getData()){
			if(d.getLabelValue() == 1){
				spamDs.addData(d);
			} else {
				noSpamDs.addData(d);
			}
		}
		
		spamDs.computeFeatureStats();
		noSpamDs.computeFeatureStats();
		
		dataSet.computeFeatureStats();
		FeatureStats fs = dataSet.getFeatureStat();
		
		for(int i = 0; i< 57; i++){
			fs.setValue(i, 5, noSpamDs.getFeatureStat().getMeanOfFeature(i));
			fs.setValue(i, 6, spamDs.getFeatureStat().getMeanOfFeature(i));
		}
		
		/********************************************/
		
		double spamCount = 0;
		double noSpamCount = 0;
		double[][] spamBucketProb = new double[57][4];
		double[][] noSpamBucketProb = new double[57][4];
		
		double[][] featureStats = fs.getSortedFeatureStats();
		
		for(int i = 0; i< dataSet.getData().size(); i++){
			Data d = dataSet.getData().get(i);
			
			if(d.getLabelValue() == 1){
				spamCount++;
				for(int f = 0; f< 57; f++){
					double val = d.getFeatureValueAtIndex(f);
					if(val >= featureStats[f][0] && val < featureStats[f][1]){
						spamBucketProb[f][0]++;
					}
					if(val >= featureStats[f][1] && val < featureStats[f][2]){
						spamBucketProb[f][1]++;
					}
					if(val >= featureStats[f][2] && val < featureStats[f][3]){
						spamBucketProb[f][2]++;
					}
					if(val >= featureStats[f][3] && val <= featureStats[f][4]){
						spamBucketProb[f][3]++;
					}
				}
			} else {
				noSpamCount++;
				for(int f = 0; f< 57; f++){
					double val = d.getFeatureValueAtIndex(f);
					if(val >= featureStats[f][0] && val < featureStats[f][1]){
						noSpamBucketProb[f][0]++;
					}
					if(val >= featureStats[f][1] && val < featureStats[f][2]){
						noSpamBucketProb[f][1]++;
					}
					if(val >= featureStats[f][2] && val < featureStats[f][3]){
						noSpamBucketProb[f][2]++;
					}
					if(val >= featureStats[f][3] && val <= featureStats[f][4]){
						noSpamBucketProb[f][3]++;
					}
				}
			}
		}
		
		for(int f = 0; f < 57; f++){
			spamBucketProb[f][0] = (spamBucketProb[f][0] + 1) / (spamCount + 2);
			spamBucketProb[f][1] = (spamBucketProb[f][1] + 1) / (spamCount + 2);
			spamBucketProb[f][2] = (spamBucketProb[f][2] + 1) / (spamCount + 2);
			spamBucketProb[f][3] = (spamBucketProb[f][3] + 1) / (spamCount + 2);
			noSpamBucketProb[f][0] = (noSpamBucketProb[f][0] + 1) / (noSpamCount + 2);
			noSpamBucketProb[f][1] = (noSpamBucketProb[f][1] + 1) / (noSpamCount + 2);
			noSpamBucketProb[f][2] = (noSpamBucketProb[f][2] + 1) / (noSpamCount + 2);
			noSpamBucketProb[f][3] = (noSpamBucketProb[f][3] + 1) / (noSpamCount + 2);
		}
		
		b4.setNoSpamProb(noSpamBucketProb);
		b4.setSpamProb(spamBucketProb);
		b4.setBuckets(featureStats);
		
		return b4;
	}

	public ErrorStat test(Bucket4 b4, DataSet testData, double thres) {
		ErrorStat err = new ErrorStat();
		testData.computeFeatureStats();
		for(int i = 0; i< testData.getDataSize(); i++){
			Data d = testData.getData().get(i);
			d.setDataSet(testData);
			double actualValue = d.getLabelValue();
			double predictedValue = computeLabel(b4, d, thres);
			if(predictedValue != actualValue){
				err.error++;
			}
			err.updateErrorRate(predictedValue, actualValue);
		}
		err.setAccuracy(testData.getDataSize());
		return err;
	}

	private double computeLabel(Bucket4 b4, Data d, double thres) {
		double probOfOne = d.getDataSet().getProbabilityOfOne();
		double probOfZero = 1 - probOfOne;
		FeatureStats fs = d.getDataSet().getFeatureStat();
		double[] x = d.getFeatureValues();
		
		b4.setSpam(false);
		double probX_Y_0 = calculateProbability(probOfZero, x, fs, b4);
		b4.setSpam(true);
		double probX_Y_1 = calculateProbability(probOfOne, x, fs, b4);
		
		double ratio = probX_Y_1 / probX_Y_0;
		rocData.add(ratio);
//		if((probX_Y_1 / probX_Y_0) >= thres){
		if(probX_Y_1 > probX_Y_0){
			return 1;
		} else {
			return 0;
		}
	}

	private double calculateProbability(double probOfY, double[] x, FeatureStats fs, Bucket4 b4) {
		double prob = 1d;
		for(int i = 0; i < x.length; i++){
			prob *= calculateValue(x[i], b4.getBuckets(), b4.getProbs(), i);
		}
		return probOfY * prob;
	}

	private double calculateValue(double val, double[][] buckets, double[][] probs, int f) {
		
		if(val >= buckets[f][0] && val < buckets[f][1]){
			return probs[f][0];
		}
		if(val >= buckets[f][1] && val < buckets[f][2]){
			return probs[f][1];
		}
		if(val >= buckets[f][2] && val < buckets[f][3]){
			return probs[f][2];
		}
		if(val >= buckets[f][3] && val <= buckets[f][4]){
			return probs[f][3];
		}
		
		return 0;
	}

}

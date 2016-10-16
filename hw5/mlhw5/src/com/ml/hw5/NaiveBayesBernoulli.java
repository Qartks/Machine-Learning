package com.ml.hw5;

import java.util.ArrayList;
import java.util.List;

public class NaiveBayesBernoulli {
	
	public List<Double> rocData = new ArrayList<Double>();

	public NBBStats train(DataSet trainingData) {
		NBBStats probStat = new NBBStats(trainingData.getFeatureSize());
		double[][] probs = new double[trainingData.getFeatureSize()][4];
		trainingData.computeFeatureStats();
		FeatureStats fs = trainingData.getFeatureStat();
		double spamCount = 0;
		double noSpamCount = 0;
		double[] spamLessMean = new double[trainingData.getFeatureSize()];
		double[] spamGrtMean = new double[trainingData.getFeatureSize()];
		double[] noSpamLessMean = new double[trainingData.getFeatureSize()];
		double[] noSpamGrtMean = new double[trainingData.getFeatureSize()];
		
		for(int i = 0; i< trainingData.getData().size(); i++){
			Data d = trainingData.getData().get(i);
			if(d.getLabelValue() == 1){
				spamCount++;
				for(int f = 0; f< d.getFeatureValues().length; f++){
					double val = d.getFeatureValueAtIndex(f);
					if(Double.isNaN(val)){
						continue;
					}
					if(val <= fs.getMeanOfFeature(f)){
						spamLessMean[f]++;
					} else {
						spamGrtMean[f]++;
					}
				}
			} else {
				noSpamCount++;
				for(int f = 0; f< d.getFeatureValues().length; f++){
					double val = d.getFeatureValueAtIndex(f);
					if(Double.isNaN(val)){
						continue;
					}
					if(val <= fs.getMeanOfFeature(f)){
						noSpamLessMean[f]++;
					} else {
						noSpamGrtMean[f]++;
					}
				}
			}
		}
		
		for(int i = 0; i< trainingData.getFeatureSize(); i++){
			double[] calcProb = new double[4];
			calcProb[0] = (spamLessMean[i] + 1)/(spamCount + 2);
			calcProb[1] = (spamGrtMean[i] + 1)/(spamCount + 2);
			calcProb[2] = (noSpamLessMean[i] + 1)/(noSpamCount + 2);
			calcProb[3] = (noSpamGrtMean[i] + 1)/(noSpamCount + 2);
			probs[i] = calcProb;
		}
		probStat.setProbStats(probs);
		return probStat;
	}

	public ErrorStat test(NBBStats nbStat, DataSet testData, double thres) {
		ErrorStat err = new ErrorStat();
		testData.computeFeatureStats();
		for(int i = 0; i< testData.getDataSize(); i++){
			Data d = testData.getData().get(i);
			d.setDataSet(testData);
			double actualValue = d.getLabelValue();
			double predictedValue = computeLabel(nbStat, d, thres);
			if(predictedValue != actualValue){
				err.error++;
			}
			err.updateErrorRate(predictedValue, actualValue);
		}
		err.setAccuracy(testData.getDataSize());
		return err;
	}

	private double computeLabel(NBBStats nbStat, Data d, double thres) {
		double probOfOne = d.getDataSet().getProbabilityOfOne();
		double probOfZero = 1 - probOfOne;
		FeatureStats fs = d.getDataSet().getFeatureStat();
		double[] x = d.getFeatureValues();
		
		nbStat.setSpam(false);
		double probX_Y_0 = calculateProbability(probOfZero, x, fs, nbStat);
		nbStat.setSpam(true);
		double probX_Y_1 = calculateProbability(probOfOne, x, fs, nbStat);
		
		double ratio = probX_Y_1 / probX_Y_0;
		rocData.add(ratio);
		
//		if((probX_Y_1 / probX_Y_0) >= thres){
		if(probX_Y_1 > probX_Y_0){
			return 1;
		} else {
			return 0;
		}
	}

	private double calculateProbability(double probY, double[] x, FeatureStats fs, NBBStats ns) {
		double prob = 1d;
		for(int i = 0; i < x.length; i++){
			if(Double.isNaN(x[i])){
				continue;
			}
			if(x[i] <= fs.getMeanOfFeature(i)){
				prob *= ns.getlessMean(i);
			} else {
				prob *= ns.getGreatMean(i);
			}
		}
		return prob * probY;
	}
	
	

}

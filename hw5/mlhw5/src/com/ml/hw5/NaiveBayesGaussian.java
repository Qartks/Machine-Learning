package com.ml.hw5;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.analysis.function.Gaussian;

public class NaiveBayesGaussian {
	
	public List<Double> rocData = new ArrayList<Double>();

	public GaussParams train(DataSet dataSet) {
		GaussParams gp = new GaussParams();
		
		DataSet spamDs = new DataSet(dataSet.getFeatureSize());
		DataSet noSpamDs = new DataSet(dataSet.getFeatureSize());
		
		for(Data d : dataSet.getData()){
			if(d.getLabelValue() == 1){
				spamDs.addData(d);
			} else {
				noSpamDs.addData(d);
			}
		}
		
		dataSet.computeFeatureStats();
		spamDs.computeFeatureStats();
		noSpamDs.computeFeatureStats();
		
		gp.setNoSpam(noSpamDs.getFeatureStat());
		gp.setSpam(spamDs.getFeatureStat());
		gp.setAllData(dataSet.getFeatureStat());
		
		gp.ProbOfSpam = (double) spamDs.getDataSize() / dataSet.getDataSize();
		return gp;
	}

	public ErrorStat test(GaussParams gp, DataSet testData, double thres) {
		ErrorStat err  = new ErrorStat();
		for(int i = 0; i< testData.getDataSize(); i++){
			Data d = testData.getData().get(i);
			d.setDataSet(testData);
			double actualValue = d.getLabelValue();
			double predictedValue = computeLabel(gp, d, thres);
			if(predictedValue != actualValue){
				err.error++;
			}
			err.updateErrorRate(predictedValue, actualValue);
		}
		err.setAccuracy(testData.getDataSize());
		return err;
	}
	
	private double computeLabel(GaussParams gp, Data d, double thres) {
		double probOfOne = gp.ProbOfSpam;
		double probOfZero = 1 - probOfOne;
		double[] x = d.getFeatureValues();
		
		double probX_Y_0 = calculateProbability(probOfZero, x, gp.getNoSpam(), gp.getAllData());
		double probX_Y_1 = calculateProbability(probOfOne, x, gp.getSpam(), gp.getAllData());
		
		double ratio = probX_Y_1 / probX_Y_0;
		rocData.add(ratio);
		if((probX_Y_1 / probX_Y_0) > thres){
//		if((probX_Y_1 > probX_Y_0)){
			return 1;
		} else {
			return 0;
		}
	}
	
	private double calculateProbability(double probY, double[] x, FeatureStats fs, FeatureStats all) {
		double prob = 1d;
		for(int i = 0; i < x.length; i++){
			prob *= caluclateGaussianValue(x[i], fs.getMeanOfFeature(i), all.getCoeffVarPerOfFeature(i));
		}
		return prob * probY;
	}

	private double caluclateGaussianValueWithApache(double x, double mean, double var) {
		double standardDeviation = Math.sqrt(var == 0 ? 0.001 : var);
		Gaussian gaussian = new Gaussian(mean, standardDeviation);
		return gaussian.value(x);
	}

	private double caluclateGaussianValue(double x, double mean, double var) {
		
		double denom = 1d / (Math.sqrt(2d * Math.PI * var));
		double mean_var = Math.pow((x - mean), 2);
		double exp = Math.exp(mean_var / var);
		
		return denom * exp;
		
	}

	private double caluclateGaussianValueWithLog(double x, double mean, double var) {
		if( var == 0){
			var = 0.001d;
		}
		double denom = Math.sqrt(2 * Math.PI * var);
		double logDenom = Math.log(denom) * -0.5;
		
		double mean_var = Math.pow((x - mean), 2);
		double exp = (-1 * mean_var) / 2 * var;
		return logDenom + exp;
	}

}

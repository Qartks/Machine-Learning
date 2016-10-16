package com.ml.hw4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class AdaBoostImplementation {
	
	double[] d; 	// Weights
	double[] π;		// Alpha values
	List<DTreeStump> ldt = new ArrayList<DTreeStump>();
	ErrorStat aL;
	ErrorStat trainErr;
	double auc;
	int rounds;
	
	public AdaBoostImplementation(DataSet trainData, int t) {
		rounds = t;
		d = new double[trainData.getDataSize()];
		π = new double[t];
		double val = 1/(double)trainData.getDataSize();
		Arrays.fill(d, val);
	}
	
	public void setAdaBoostImplementation(double[] id, double[] iπ) {
		d = id;
		π = iπ;
	}

	public void train(DataSet trainData, DataSet testData, int t, boolean optimal) throws CloneNotSupportedException {
		
		DTreeStump dStump = new DTreeStump(trainData, d, optimal);
//		int i = 0;
//		while(ldt.size() < t){
		for(int i = 0; i< t; i++){
			
			getWeakLearnerHypothesis(d, trainData, optimal, dStump);
			
			DTreeStump dd = (DTreeStump) dStump.clone();
			
			double epsilon = dStump.getEpsilon();
			ldt.add(dd);
//			if(Math.abs(epsilon - 0.5) > 0.001){
//				ldt.add(dd);
//			} else {
//				continue;
//			}
			
			double alpha = calculateAlpha(epsilon);
			π[i] = alpha;
			updateWeights(dd, trainData, i);
			
			ErrorStat errorTrain = test(trainData, i);
			trainErr = errorTrain;
			ErrorStat errorTest =  new ErrorStat();
			if(testData != null){
				errorTest = test(testData, i);
				
				aL = errorTest;
				ErrorStat aucError;
				Collections.sort(errorTest.thresVals);
				
				List<Double> tpr = new ArrayList<Double>();
				List<Double> fpr = new ArrayList<Double>();
				
				for(double thres : errorTest.thresVals){
					aucError = getAUC(testData, i, thres);
					tpr.add(aucError.getTPRate());
					fpr.add(aucError.getFPRate());
				}
				
				auc = calculateAUC(fpr,tpr);
			}
//			i++;
//			System.out.println((i+1) + "\t" + epsilon + " \t" + errorTrain.error + "\t" + errorTest.error + "\t" + auc);
//			System.out.println("Round " + (i+1) + " -> "+ (dStump.featureIndex+1) + " -> " + dStump.thresholdValue + " -> Round error:" + epsilon + " -> Train Error: " + errorTrain.error + " -> Test Error: " + errorTest.error + " -> AUC: " + auc) ;
//			System.out.println("Round " + (i+1) + " -> "+ (dStump.featureIndex+1) + " -> " + dStump.thresholdValue + " -> Round error:" + epsilon + " -> Train Error: " + errorTrain.error + " -> Test Error: " + errorTest.error + " -> AUC: " + auc);
		}
//		System.out.println(Arrays.toString(π));
		
	}

	private double calculateAUC(List<Double> fpr, List<Double> tpr) {
		double auc = 0;
		for(int i = 1; i<fpr.size(); i++){
			double tpr1 = tpr.get(i);
			double tpr2 = tpr.get(i-1);
			double fpr1 = fpr.get(i);
			double fpr2 = fpr.get(i-1);
			
			auc += (fpr2 - fpr1)*(tpr1 + tpr2);
		}
		return auc * 0.5;
	}

	private void updateWeights(DTreeStump dStump, DataSet trainData, int t) {
		double z = 0;
		for(int i = 0; i< trainData.getDataSize(); i++){
			Data data = trainData.getData().get(i);
			double hi = dStump.getPredicted(data);
			double yi = data.getLabelValue() == 0 ? -1 : 1;
//			System.out.println(hi + " -> " + yi);
			d[i] = d[i] * Math.exp(-1 * π[t] * hi * yi);
			z += d[i];
		}
		
		for(int i=0; i<trainData.getDataSize(); i++){
			d[i] /= z;
		}
	}

	private double calculateAlpha(double e) {
		double val = (1 - e)/e;
		double val2 = Math.log(val);
		return 0.5 * val2;
	}

	private void getWeakLearnerHypothesis(double[] d, DataSet trainData, boolean optimal, DTreeStump dt) {	
		
		if(!optimal){
			dt.getRandomStump();
		} else {
			dt.generateBestStump();
		}
	}

	public ErrorStat test(DataSet dataSet, int t) {
		ErrorStat err = new ErrorStat();
		List<Double> tVals = new ArrayList<Double>();
		List<Integer> tInd = new ArrayList<Integer>();
		for(Data data : dataSet.getData()){
			tInd.add(dataSet.getData().indexOf(data));
			double predictedValue = getPred(t, data, tVals); 
			double actualValue = data.getLabelValue() == 0 ? -1 : 1;
			if(predictedValue != actualValue){
				err.error++;
			}
			err.updateErrorRate(predictedValue, actualValue);
		}
		err.error /= dataSet.getDataSize();
		err.thresVals = tVals;
		err.indVals = tInd;
		return err;
	}

	private double getPred(int t, Data data, List<Double> tVals) {
		double sum = 0;
		for(int i = 0; i < t+1; i++){
			DTreeStump model = ldt.get(i);
			sum += (π[i] * model.getPredicted(data));
		}
		tVals.add(sum);
		if(sum >= 0){
			return 1;
		}
		return -1;
	}
	
	private ErrorStat getAUC(DataSet dataSet, int t, double thres) {
		ErrorStat err = new ErrorStat();
		for(Data data : dataSet.getData()){			
			double predictedValue = getPredAUC(t, data, thres);
			double actualValue = data.getLabelValue() == 0 ? -1 : 1;
			err.updateErrorRate(predictedValue, actualValue);
		}
		return err;
	}

	private double getPredAUC(int t, Data data, double thres) {
		double sum = 0;
		for(int i = 0; i < t; i++){
			DTreeStump model = ldt.get(i);
			sum += (π[i] * model.getPredicted(data));
		}
		
		if(sum >= thres){
			return 1;
		}
		return -1;
	}

	public double predictValue(Data data) {
		double sum = 0;
		for(int i = 0; i < rounds; i++){
			DTreeStump model = ldt.get(i);
			sum += (π[i] * model.getPredicted(data));
		}
		
		if(sum >= 0){
			return 1;
		}
		return 0;
	}

}

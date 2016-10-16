package com.ml.hw4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ErrorStat {

	double error;
	double[] errR;
	double accuracy;
	public double metaPred;
	public List<Double> thresVals;
	public List<Integer> indVals;
	
	public ErrorStat() {
		error = 0;
		errR = new double[4];
		accuracy = 0;
		thresVals = new ArrayList<Double>();
	}

	public double getAccuracy() {
		return accuracy;
	}
	
	public void setAccuracy(int dataSize){
		 accuracy = 100d - (error/dataSize * 100);
	}
	
	public double getTPRate(){
		return (errR[0] + 0)/(errR[0] + errR[1] + 0);
	}
	
	public double getFPRate(){
		return (errR[2] + 0)/(errR[2] + errR[3] + 0);
	}
	
	public void updateErrorRate(double pV, double aV){
		
		if(pV == 1 && aV == 1){ // True Positive
			errR[0]++;
		}
		if(pV == -1 && aV == 1){ // False Negative
			errR[1]++;
		}
		if(pV == 1 && aV == -1){ // False Positive
			errR[2]++;
		}
		if(pV == -1 && aV == -1){ // True Negative
			errR[3]++;
		}
	}
	
	public String toString(){
		return Arrays.toString(errR);
	}

	public void incrementErrorRates(double[] er) {
		for(int i = 0; i< 4; i++){
			errR[i] += er[i];
		}
	}
}

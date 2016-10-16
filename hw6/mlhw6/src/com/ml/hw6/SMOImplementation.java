package com.ml.hw6;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class SMOImplementation {
	
	double C;
	double epsilon;
	double tol;
	int maxIterations;
	DataSet dataSet;
	double[] alpha;
	double b = 0;
	Map<myPair,Double> cache;
	
	public SMOImplementation(double iC, double iE, double iTol, DataSet dS, int iter) {
		this.C = iC;
		this.epsilon = iE;
		this.tol = iTol;
		this.dataSet = dS;
		this.maxIterations = iter;
		this.alpha = new double[dS.getDataSize()];
		this.cache = new HashMap<myPair, Double>();
	}

	public void buildLagrangeMultipliers() {
		int numChanged = 0;
		boolean examineAll = true;
		
		while(numChanged > 0 | examineAll){
			System.out.println(Arrays.toString(alpha));
			numChanged = 0;
			if(examineAll){
				for(int i = 0; i < dataSet.getDataSize(); i++){
					numChanged += examineExample(i);
				}
			} else {
				List<Integer> list = getFilteredList();
				for(int i : list){
					numChanged += examineExample(i);
				}
			}
			if(examineAll){
				examineAll = false;
			} else if(numChanged == 0){
				examineAll = true;
			}
		}
	}

	private int examineExample(int i) {
		double yi = dataSet.getData().get(i).labelValue == 0 ? -1 : 1;
		double alphai = alpha[i];
		double fx = getSVMoutPut(i);
		double Ei = fx - yi;
		double ri = Ei * yi;
		
		if ((ri < -tol && alphai < C) || (ri > tol && alphai > 0)){
			List<Integer> check_indices = new ArrayList<Integer>();
			if(getNumberOfNonZeroNonCAlphas() > 1){
				int j = selectRandomInotEqualJ(i);
				if(takeStep(i, j)){
					return 1;
				}
				check_indices.add(j);
			}
			List<Integer> list = getFilteredList();
			Collections.shuffle(list);
			for(int j : list){
				if(check_indices.contains(j)){
					continue;
				}
				if(takeStep(i, j)){
					return 1;
				}
				check_indices.add(j);
			}
			List<Integer> randList = getRandomList();
			for(int j : randList){
				if(check_indices.contains(j)){
					continue;
				}
				if(takeStep(i, j)){
					return 1;
				}
			}
		}
		return 0;
	}
	
	private List<Integer> getRandomList() {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0 ; i < dataSet.getDataSize(); i++){
			list.add(i);
		}
		
		Collections.shuffle(list);
		return list;
	}

	private int selectRandomInotEqualJ(int i) {
		int size = dataSet.getDataSize();
		int j = i;
		while(i == j){
			Random rand = new Random();
			j = rand.nextInt(size);
		}
		return j;
	}
	
	private double getSVMoutPut(int i2) {
		double output = 0; 
		for(int i = 0; i < dataSet.getDataSize(); i++){
			double val = 0;
			myPair p = new myPair(i, i2);
			if(cache.containsKey(p)){
				val = alpha[i] * dataSet.getData().get(i).labelValue * cache.get(p);
			} else {
				double k = kernel(dataSet.getData().get(i), dataSet.getData().get(i2), i, i2);
				cache.put(p, k);
				val = alpha[i] * dataSet.getData().get(i).labelValue * k;
			}
			output += val;
		}
		return output + b;
	}

	private boolean takeStep(int i, int j) {
		if(i == j){
			return false;
		}
		double alphai = alpha[i];
		double alphaj = alpha[j];
		double yi = dataSet.getData().get(i).labelValue  == 0 ? -1 : 1;
		double yj = dataSet.getData().get(j).labelValue  == 0 ? -1 : 1;
		double fxi = getSVMoutPut(i);
		double Ei = fxi - yi;
		double fxj = getSVMoutPut(j);
		double Ej = fxj - yj;
		double s = yi * yj;
		
		Data di = dataSet.getData().get(i);
		Data dj = dataSet.getData().get(j);
		
		double L = 0;
		double H = 0;
		if (yi != yj){
            L = Math.max(0, alphaj - alphai);
            H = Math.min(C, alphaj - alphai + C);
		} else {
            L = Math.max(0, alphai + alphaj - C);
            H = Math.min(C, alphai + alphaj);
		}
		
		if(L == H){
			return false;
		}
		double k11 = kernel(di, di, i, j);
		double k12 = kernel(di, dj, i, j);
		double k22 = kernel(dj, dj, i, j);
		double eta = k11 + k22 - 2*k12;
		
		if (eta <= 0){
			return false;
		}
		
		double alphaj_new = alphaj + yj * (Ei - Ej) / eta;
		double alphaj_new_clipped = 0;
		if (alphaj_new < L){
	           alphaj_new_clipped = L;
		} else if(L <= alphaj_new && alphaj_new <= H){
	           alphaj_new_clipped = alphaj_new;
		} else {
	           alphaj_new_clipped = H;
		}
	    alphaj_new = alphaj_new_clipped;
		
	    if (Math.abs(alphaj - alphaj_new) < epsilon * (alphaj + alphaj_new + epsilon)){
            return false;
	    }

	    double alphai_new = alphai + s * (alphaj - alphaj_new);
        alpha[i] = alphai_new;
        alpha[j] = alphaj_new;
        
        double b_i_new = b - Ei + (alphai - alphai_new) * yi * k11 + (alphaj - alphaj_new) * yj * k12;
        double b_j_new = b - Ej + (alphai - alphai_new) * yi * k12 + (alphaj - alphaj_new) * yj * k22;
        double b_new = 0;
        if (0 < alphai_new && alphai_new < C){
            b_new = b_i_new;
        } else if (0 < alphaj_new && alphaj_new < C){
        	b_new = b_j_new;
        } else {
        	b_new = (b_i_new + b_j_new) / 2;
        }
        b = b_new;
        
		return true;
	}

	private double kernel(Data data, Data data2, int i, int j) {
		double sum = 0;
		myPair p = new myPair(i, j);
		if(cache.containsKey(p)){
			return cache.get(p);
		} else {
			double[] f1 = data.featureValues;
			double[] f2 = data2.featureValues;
			for(int k = 0; k < dataSet.getFeatureSize(); k++){
				sum += f1[k] * f2[k];
			}
			cache.put(p, sum);
		}
		return sum;
	}

	private int getNumberOfNonZeroNonCAlphas() {
		return getFilteredList().size();
	}

	private List<Integer> getFilteredList() {
		List<Integer> list = new ArrayList<Integer>();
		for(int i = 0; i < dataSet.getDataSize(); i++){
			if(alpha[i] > 0 && alpha[i] < C){
				list.add(i);
			}
		}
		return list;
	}
	
	public void predictTrain() {
		double error = 0;
		for(int i = 0; i < dataSet.getDataSize(); i++){
			Data d = dataSet.getData().get(i);
			double actualValue = d.getLabelValue();
			double predictedValue = getPredictedValue(i,d);
			if(actualValue != predictedValue){
				error++;
			}
		}
		System.out.println("Accuracy : " + (error / dataSet.getDataSize()));
	}

	private double getPredictedValue(int i, Data d) {
		double sum = 0;
		double yi = d.labelValue; 
		for(int j = 0; j < dataSet.getDataSize(); j++){
			myPair p = new myPair(i, j); 
			double k = 0;
			if(cache.containsKey(p)){
				k = cache.get(p);
			} else {
				k = kernel(dataSet.getData().get(i), dataSet.getData().get(j),i,j);
				cache.put(p, k);
			}
			sum += (alpha[j] * yi * k);
		}
		return Math.signum(sum + b);
	}

	public void predictTest(DataSet x) {
		double error = 0;
		for(int i = 0; i < x.getDataSize(); i++){
			Data d = x.getData().get(i);
			double actualValue = d.getLabelValue();
			double predictedValue = getPredictedValue(i,d);
			if(actualValue != predictedValue){
				error++;
			}
		}
		System.out.println("Accuracy : " + (error / x.getDataSize()));
	}

}

package com.ml.hw6;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class SMOSimplified {
	
	double C;
	double epsilon;
	double tol;
	int maxIterations;
	DataSet dataSet;
	double[] alpha;
	Map<Integer,Double> Fx;
	double b = 0;
	Map<myPair,Double> cache;
	double[][] cache2;
	
	public SMOSimplified(double iC, double iE, double iTol, DataSet dS, int iter) {
		this.C = iC;
		this.epsilon = iE;
		this.tol = iTol;
		this.dataSet = dS;
		this.maxIterations = iter;
		this.alpha = new double[dS.getDataSize()];
		this.Fx = new HashMap<Integer,Double>();
		this.cache = new HashMap<myPair, Double>();
		cache2 = new double[dS.getDataSize()][dS.getDataSize()];
	}

	public void buildLagrangeMultipliers() {
		int passes = 0;
		int counter = 0;
		
		while(passes < maxIterations){
			int num_changed_alphas = 0;
			for(int i = 0; i < dataSet.getDataSize(); i++){
				double yi = dataSet.getData().get(i).getLabelValue() == 0 ? -1 : 1;
				
				double fi = 0;
				if(Fx.containsKey(i)){
					fi = Fx.get(i);
				} else {
					fi = getSVMoutPut(i);
					Fx.put(i, fi);
				}
				double Ei = fi - yi;
				
				double alphai = alpha[i];
				if((yi*Ei < -tol && alphai < C) || (yi*Ei > tol && alphai > 0)){
					
					int j = selectRandomInotEqualJ(i);
					
					double yj = dataSet.getData().get(j).getLabelValue() == 0 ? -1 : 1;
					
					double fj = 0;
					if(Fx.containsKey(j)){
						fj = Fx.get(j);
					} else {
						fj = getSVMoutPut(j);
						Fx.put(j, fj);
					}
					double Ej = fj - yj;
					
					double alphaIOld = alpha[i];
					double alphaJOld = alpha[j];
					double oldB = b;
					
					double L = computeL(yi, yj,i ,j);
					double H = computeH(yi, yj,i ,j);
					
					if(L == H){
						continue;
					}
					double eta = computeEta(i,j);
					if(eta >= 0){
						continue;
					}
					alpha[j] = compluteAlphaJClip(Ei,Ej,eta,j,yj,L,H,alphaJOld);
					
					if(Math.abs(alpha[j] - alphaJOld) < 0.00001){
						updateFxNew(i, alphaIOld, j, alphaJOld, oldB);
						continue;
					}
					alpha[i] = alpha[i] + yi*yj*(alphaJOld - alpha[j]);
					
					double b1 = computeB1(i, j, Ei, yi, alphaIOld, yj, alphaJOld);
					double b2 = computeB2(i, j, Ej, yi, alphaIOld, yj, alphaJOld);
					b = computeB(b1, b2,i, j);
					updateFxNew(i, alphaIOld, j, alphaJOld, oldB);
					num_changed_alphas = num_changed_alphas + 1;
				}
			}
//			System.out.println(Arrays.toString(alpha));
			System.out.println(passes);
			System.out.println(num_changed_alphas);
			counter++;
			if(num_changed_alphas == 0){
				passes = passes + 1;
			} else {
				passes = 0;
			}
			if(counter > 300){
				break;
			}
		}
	}
	
	private void updateFxNew(int i, double alphaIOld, int j, double alphaJOld, double oldB) {
		Data dj = dataSet.getData().get(j);
		Data di = dataSet.getData().get(i);
		double yi = di.labelValue== 0 ? -1 : 1;
		double yj = dj.labelValue== 0 ? -1 : 1;
		
		for(int k = 0 ; k < dataSet.getDataSize(); k++){
			
			if(Fx.containsKey(k)){
				double value = Fx.get(k);
				Data dk = dataSet.getData().get(k);
				double yk = dk.labelValue == 0 ? -1 : 1;
				
				myPair p = new myPair(k, i);
				double simi = cache.get(p);
				
				myPair p2 = new myPair(k, j);
				double simi2 = cache.get(p2);
				
//				value += (alpha[i] - alphaIOld) * yi * simi;
//				value += ((alpha[j] - alphaJOld) * yj * simi2);
//				value += (b - oldB);
//				Fx.put(k, value);
				
				Fx.put(k, value + ((alpha[i] - alphaIOld) * yi * simi) + ((alpha[j] - alphaJOld) * yj * simi2) + (b - oldB));
			}
		}
	}
	

	private double getSVMoutPut(int j) {
		double output = 0; 
		Data dj = dataSet.getData().get(j);
		for(int i = 0; i < dataSet.getDataSize(); i++){
			Data di = dataSet.getData().get(i);
			double val = 0;
			double yi = di.labelValue== 0 ? -1 : 1; 
			
			myPair p = new myPair(i, j);
			if(cache.containsKey(p)){
				val = alpha[i] * yi * cache.get(p);
			} else {
				double k = kernel(di, dj);
				cache.put(p, k);
				val = alpha[i] * yi * k;
			}
			
			output += val;
		}
		return output + b;
	}

	private double computeB2(int i, int j, double ej, double yi, double alphaIOld, double yj, double alphaJOld) {
		
		double k11 = 0;
		double k12 = 0;
		
		Data di = dataSet.getData().get(i);
		Data dj = dataSet.getData().get(j);
		
		myPair p = new myPair(j, j);
		if(cache.containsKey(p)){
			k11 = cache.get(p);
		} else {
			k11 = kernel(dj, dj);
			cache.put(p, k11);
		}
		
		myPair p2 = new myPair(i, j);
		if(cache.containsKey(p2)){
			k12 = cache.get(p2);
		} else {
			k12 = kernel(di, dj);
			cache.put(p2, k12);
		}
		
//		k11 = kernel(dataSet.getData().get(j), dataSet.getData().get(j));
//		k12 = kernel(dataSet.getData().get(i), dataSet.getData().get(j));
		
		double a1 = alpha[i] - alphaIOld;
		double val1 = yi * a1 * k12;
		double a2 = alpha[j] - alphaJOld;
		double val2 = yj * a2 * k11;
		
		double b2 = b - ej;
		b2 -= val1;
		b2 -= val2;
		return b2;
	}

	private double computeB1(int i, int j, double ei, double yi, double alphaIOld, double yj, double alphaJOld) {
//		double k11 = kernel(dataSet.getData().get(i), dataSet.getData().get(i));
//		double k12 = kernel(dataSet.getData().get(i), dataSet.getData().get(j));
		
		double k11 = 0;
		double k12 = 0;
		
		Data di = dataSet.getData().get(i);
		Data dj = dataSet.getData().get(j);
		
		myPair p = new myPair(i, i);
		if(cache.containsKey(p)){
			k11 = cache.get(p);
		} else {
			k11 = kernel(di, di);
			cache.put(p, k11);
		}
		
		myPair p2 = new myPair(i, j);
		if(cache.containsKey(p2)){
			k12 = cache.get(p2);
		} else {
			k12 = kernel(di, dj);
			cache.put(p2, k12);
		}
		
		double a1 = alpha[i] - alphaIOld;
		double val1 =  yi * k11 * a1;
		double a2 = alpha[j] - alphaJOld;
		double val2 =  yj * k12 * a2;
		
		double b1 = b - ei;
		b1 -= val1;
		b1 -= val2;
		return b1;
	}

	private double computeB(double b1, double b2, int i, int j) {
		double val = 0;
		if(alpha[i] > 0 && alpha[i] < C){
			val = b1;
		} else if(alpha[j] > 0 && alpha[j] < C){
			val = b2;
		} else {
			val = (b1 + b2)/2;
		}
		
		return val;
	}

	private double compluteAlphaJClip(double ei, double ej, double eta, int j, double yj, double L, double H, double oldAlphaJ) {
		double val = yj * (ei - ej)/eta;
		double val2 = oldAlphaJ - val;
		
		if(val2 < L){
			return L;
		} else if(val2 > H){
			return H;
		}
		
		return val2;
	}

	private double computeEta(int i, int j) {
//		double k11 = kernel(dataSet.getData().get(i), dataSet.getData().get(i));
//		double k12 = kernel(dataSet.getData().get(i), dataSet.getData().get(j));
//		double k22 = kernel(dataSet.getData().get(j), dataSet.getData().get(j));
		
		double k11 = 0;
		double k12 = 0;
		double k22 = 0;
		
		Data di = dataSet.getData().get(i);
		Data dj = dataSet.getData().get(j);
		
		myPair p = new myPair(j, j);
		if(cache.containsKey(p)){
			k22 = cache.get(p);
		} else {
			k22 = kernel(dj, dj);
			cache.put(p, k22);
		}
		
		myPair p2 = new myPair(i, j);
		if(cache.containsKey(p2)){
			k12 = cache.get(p2);
		} else {
			k12 = kernel(di, dj);
			cache.put(p2, k12);
		}
		
		myPair p3 = new myPair(i, i);
		if(cache.containsKey(p3)){
			k11 = cache.get(p3);
		} else {
			k11 = kernel(di, di);
			cache.put(p3, k11);
		}
		
		double eta = (2 * k12) - k11 - k22;
		return eta;
	}

	private double kernel(Data data, Data data2) {
		double[] f1 = data.featureValues;
		double[] f2 = data2.featureValues;
		
		double sum = 0;
		for(int i = 0; i < dataSet.getFeatureSize(); i++){
			sum += f1[i] * f2[i];
		}
		return sum;
	}

	private double computeH(double yi, double yj, int i, int j) {
		if(yi == yj){
			return Math.min(C, alpha[i]+alpha[j]);
		} else {
			return Math.min(C, C + alpha[j] - alpha[i]);
		}
	}

	private double computeL(double yi, double yj, int i, int j) {
		if(yi == yj){
			return Math.max(0, alpha[i]+alpha[j] - C);
		} else {
			return Math.max(0, alpha[j] - alpha[i]);
		}
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

	public double predictTrain() {
		double error = 0;
		for(int i = 0; i < dataSet.getDataSize(); i++){
			Data d = dataSet.getData().get(i);
			double actualValue = d.getLabelValue() == 0 ? -1 : 1;
//			double actualValue = d.getLabelValue();
			double predictedValue = getPredictedValue(i,d, dataSet);
			if(actualValue != predictedValue){
				error++;
			}
		}
		double acc = ((1 - error / dataSet.getDataSize())*100);
		System.out.println("Train Accuracy : " + acc);
		return acc;
	}

	private double getPredictedValue(int i, Data d, DataSet ds) {
		double sum = 0;
		double yi = d.labelValue== 0 ? -1 : 1; 
		for(int j = 0; j < dataSet.getDataSize(); j++){
			double yj = dataSet.getData().get(j).labelValue== 0 ? -1 : 1; 
			if(alpha[j] == 0){
				continue;
			}
			myPair p = new myPair(i, j); 
			double k = kernel(d, dataSet.getData().get(j));
			sum += (alpha[j] * yj * k);
		}
		sum = sum + b;
		
		if(sum > 0){
			return 1;
		} else {
			return -1;
		}
//		return Math.signum(sum);
	}

	public double predictTest(DataSet x) {
		double error = 0;
		for(int i = 0; i < x.getDataSize(); i++){
			Data d = x.getData().get(i);
			double actualValue = d.getLabelValue() == 0? -1 : 1;
//			double actualValue = d.getLabelValue();
			double predictedValue = getPredictedTestValue(i,d ,x);
			if(actualValue != predictedValue){
				error++;
			}
		}
		double acc = ((1 - error / x.getDataSize())*100);
		System.out.println("Test Accuracy : " + acc);
		return acc;
	}
	
	private double getPredictedTestValue(int i, Data d, DataSet ds) {
		double sum = 0;
		double y = d.labelValue== 0 ? -1 : 1; 
		for(int j = 0; j < dataSet.getDataSize(); j++){
			Data dj = dataSet.getData().get(j);
			double yj = dj.labelValue== 0 ? -1 : 1; 
			if(alpha[j] == 0){
				continue;
			}
			double k = 0;
			k = kernel(d, dj);
			sum += (alpha[j] * yj * k);
		}
		sum = sum + b;
		
		if(sum > 0){
			return 1;
		} else {
			return -1;
		}
//		return Math.signum(sum);
	}

	public double getFXvalue(Data d) {
		double sum = 0;
		for(int j = 0; j < dataSet.getDataSize(); j++){
			if(alpha[j] == 0){
				continue;
			}
			Data dj = dataSet.getData().get(j);
			double yj = d.labelValue== 0 ? -1 : 1; 
			double k = 0;
			k = kernel(d, dj);
			sum += (alpha[j] * yj * k);
		}
		sum = sum + b;
		return sum;
	}
}

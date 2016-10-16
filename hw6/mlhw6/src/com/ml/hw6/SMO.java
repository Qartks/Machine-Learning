package com.ml.hw6;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class SMO {
	
	double C;
	double epsilon;
	double tol;
	int maxIterations;
	DataSet dataSet;
	double[] alphas;
	Map<Integer,Double> Fx;
	double b = 0;
	double[][] cache;
	String KERNEL_TYPE = "linear";
	
	public SMO(double iC, double iE, double iTol, DataSet dS, int iter) {
		this.C = iC;
		this.epsilon = iE;
		this.tol = iTol;
		this.dataSet = dS;
		this.maxIterations = iter;
		this.alphas = new double[dS.getDataSize()];
		this.Fx = new HashMap<Integer,Double>();
		this.cache = new double[dS.getDataSize()][dS.getDataSize()];
		
		for(int i = 0 ; i < dS.getDataSize(); i++){
			for(int j = 0 ; j < dS.getDataSize(); j++){
				cache[i][j] = Double.NaN;
			}
		}
		
	}

	public void buildLagrangeMultipliers() {
		int passes = 0;
		int counter = 0;
		while(passes < maxIterations){
			int num_alphas_changed = 0;
			for(int i = 0; i < dataSet.getDataSize(); i++){
				Data di = dataSet.getData().get(i);
				double yi = di.labelValue == 0 ? -1 : 1;
				double fi = calculateFx(i);
				double Ei = fi - yi;
				double alphai = alphas[i];
				if(((yi*Ei < -tol) && (alphai < C)) || ((yi*Ei > tol) && (alphai > 0))){
					int j = selectRandomInotEqualJ(i);
					Data dj = dataSet.getData().get(j);
					double yj = dj.labelValue == 0 ? -1 : 1;
					double fj = calculateFx(j);
					double Ej = fj - yj;
					double alphaIOld = alphas[i];
					double alphaJOld = alphas[j];
					double bOld = b;
					double L = computeL(yi, yj, i, j);
					double H = computeH(yi, yj, i, j);
					if(L == H){
						continue;
					}
					double eta = computeEta(i, j);
					if(eta >= 0){
						continue;
					}
					double alphaJ = calculateAlphaJ(alphas[j], yj, Ei, Ej, eta, L, H);
					alphas[j] = alphaJ;
					if(Math.abs(alphas[j] - alphaJOld) < .00001) {
						updateFXCache(i, j, alphaIOld, alphaJOld, bOld);
						continue;
					}
					alphas[i] = alphas[i] + yi*yj*(alphaJOld - alphas[j]);
					double b1 = computeB1(Ei, yi, yj, alphaIOld, alphaJOld, i, j);
					double b2 = computeB2(Ej, yi, yj, alphaIOld, alphaJOld, i, j);
					computeB(b1, b2, i, j);
					updateFXCache(i, j, alphaIOld, alphaJOld, bOld);
					num_alphas_changed += 1;
				}	
			}
			System.out.println(passes);
			System.out.println(num_alphas_changed);
			counter++;
			if(num_alphas_changed == 0){
				passes = passes + 1;
			} else {
				passes = 0;
			}
			if(counter > 300){
				break;
			}
		}
		
	}
	
	private void computeB(double b1, double b2, int i, int j) {
		if(alphas[i] > 0 && alphas[i] < C){
			b = b1;
		} else if(alphas[j] > 0 && alphas[j] < C){
			b = b2;
		} else {
			b = (b1 + b2)/2;
		}
	}

	private double computeB2(double ej, double yi, double yj, double alphaIOld, double alphaJOld, int i, int j) {
		Data dI = dataSet.getData().get(i);
		Data dJ = dataSet.getData().get(j);
		double alphaX1 = alphas[i];
		double alphaX2 = alphas[j];
//		double b2 = b - ej - (yi * (alphaX1 - alphaIOld) * kernel(i, j, dI, dJ, true)) - (yj * (alphaX2 - alphaJOld) * kernel(j, j, dJ, dJ, true));
		double b2 = b - ej; 
		b2 -= (yi * (alphaX1 - alphaIOld) * kernel(i, j, dI, dJ, true));
		b2 -= (yj * (alphaX2 - alphaJOld) * kernel(j, j, dJ, dJ, true));
		return b2;
	}

	private double computeB1(double ei, double yi, double yj, double alphaIOld, double alphaJOld, int i, int j) {
		Data dI = dataSet.getData().get(i);
		Data dJ = dataSet.getData().get(j);
		double alphaX1 = alphas[i];
		double alphaX2 = alphas[j];
//		double b1 = b - ei - (yi * (alphaX1 - alphaIOld) * kernel(i, i, dI, dI, true)) - (yj * (alphaX2 - alphaJOld) * kernel(i, j, dI, dJ, true));
		double b1 = b - ei;
		b1 -= (yi * (alphaX1 - alphaIOld) * kernel(i, i, dI, dI, true));
		b1 -= (yj * (alphaX2 - alphaJOld) * kernel(i, j, dI, dJ, true));
		return b1;
	}

	private void updateFXCache(int i, int j, double alphaIOld, double alphaJOld, double bOld) {
		Data dI = dataSet.getData().get(i);
		Data dJ = dataSet.getData().get(j);
		double yI = dI.labelValue == 0 ? -1 : 1;
		double yJ = dJ.labelValue == 0 ? -1 : 1;
		for(int k = 0; k < dataSet.getDataSize(); k++) {
			Data dK = dataSet.getData().get(k);
			if(Fx.containsKey(k)) {
				double value = Fx.get(k);
				value+= (alphas[i] - alphaIOld) * yI * kernel(k, i, dK, dI, true);
				value+= (alphas[j] - alphaJOld) * yJ * kernel(k, j, dK, dJ, true);
				value+= (b - bOld);
				Fx.put(k, value);
				
//				double simi = kernel(i, k, dI, dK, true);
//				double simi2 = kernel(j, k, dJ, dK, true);
//				Fx.put(k, value + ((alphas[i] - alphaIOld) * yI * simi) + ((alphas[j] - alphaJOld) * yJ * simi2) + (b - bOld));
			}
		}
	}

	private double calculateAlphaJ(double aJ, double yJ, double ei, double ej, double eta, double L, double H) {
		double val = yJ * (ei - ej)/eta;
		double alphaJ = aJ - val;
		
		if(alphaJ > H) {
			alphaJ = H;
		}
		if(alphaJ < L) {
			alphaJ = L;
		}
		
		return alphaJ;
	}

	private double computeEta(int i, int j) {
		Data dI = dataSet.getData().get(i);
		Data dJ = dataSet.getData().get(j);
		double kIJ = kernel(i, j, dI, dJ, true);
		double kII = kernel(i, i, dI, dI, true);
		double kJJ = kernel(j, j, dJ, dJ, true);
		double eta = (2 * kIJ) - kII - kJJ;
		return eta;
	}

	private double computeH(double yi, double yj, int i, int j) {
		if(yi == yj){
			return Math.min(C, alphas[i]+alphas[j]);
		} else {
			return Math.min(C, C + alphas[j] - alphas[i]);
		}
	}

	private double computeL(double yi, double yj, int i, int j) {
		if(yi == yj){
			return Math.max(0, alphas[i]+alphas[j] - C);
		} else {
			return Math.max(0, alphas[j] - alphas[i]);
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

	private double calculateFx(int i) {
		double value = 0;
		if(Fx.containsKey(i)) {
			value = Fx.get(i);
		} else {
			value = computeFX(i);
			Fx.put(i, value);
		}
		return value;
	}

	private double computeFX(int i) {
		double value = 0;
		Data dataI = dataSet.getData().get(i);
		for(int j = 0; j < dataSet.getData().size(); j++) {
			Data dataJ = dataSet.getData().get(j);
			double yJ = dataJ.labelValue == 0 ? -1 : 1;
			value += alphas[j] * yJ * kernel(i, j, dataI, dataJ, true);
		}
		value+= b;
		return value;
	}

	private double kernel(int i, int j, Data dataI, Data dataJ, boolean useCache) {
		double k = Double.NaN;
		k = cache[i][j];
		if(Double.isNaN(k)) {
			k = evaluateDotProduct(dataI, dataJ);
			cache[i][j] = k;
			cache[j][i] = k;
		} 
		return k;
	}
	
	private double evaluateDotProduct(Data d1, Data d2) {
		double[] f1 = d1.featureValues;
		double[] f2 = d2.featureValues;
		
		double sum = 0;
		for(int i = 0; i < dataSet.getFeatureSize(); i++){
			sum += f1[i] * f2[i];
		}
		return sum;
	}
	

	public double predict(DataSet t) {
		double error = 0;
		for(int i = 0 ; i < t.getDataSize(); i++){
			Data d = t.getData().get(i);
			double actualLabel = d.labelValue == 0 ? -1 : 1;
			double predictedLabel = predictedValue(d);
			if(actualLabel != predictedLabel){
				error++;
			}
		}
		double acc = (1 - (error / t.getDataSize())) * 100;
		return acc;
	}

	private double predictedValue(Data d) {
		double fx = 0;
		for(int i = 0 ; i < dataSet.getDataSize(); i++){
			if(alphas[i] == 0){
				continue;
			}
			Data dT = dataSet.getData().get(i);
			double dY = dT.labelValue == 0 ? -1 : 1;
			double k = evaluateDotProduct(dT, d);
			fx += (alphas[i] * dY * k);
		}
		fx = fx + b;
		return fx >= 0 ? 1 : -1;
	}

	public double getFXvalue(Data d) {
		double value = 0;
		for(int j = 0; j < dataSet.getData().size(); j++) {
			Data dataJ = dataSet.getData().get(j);
			double yJ = dataJ.labelValue == 0 ? -1 : 1;
			value += alphas[j] * yJ * evaluateDotProduct(d, dataJ);
		}
		value+= b;
		return value;
	}

	public void setTrainingLabels(double dig, List<Double> labelList) {
		for(int i = 0 ; i < dataSet.getDataSize(); i++){
			Data d = dataSet.getData().get(i);
			double lab = labelList.get(i);
			if(lab == dig){
				d.setLabelValue(1d);
			} else {
				d.setLabelValue(0d);
			}
		}
		
	}

	
}

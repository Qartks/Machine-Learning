package com.ml.hw6;

public class EuclidDistance extends Kernel {

	public EuclidDistance(int size) {
		super(size);
	}

	@Override
	public double evaluate(Data d1, Data d2) {
		double[] f1 = d1.featureValues;
		double[] f2 = d2.featureValues;
		
		double sum = 0;
		for(int i = 0; i < d1.featureValues.length; i++){
			sum += f1[i] * f2[i];
		}
		return sum;
	}

}

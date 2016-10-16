package com.ml.hw7;

public class EuclidDistance extends Kernel {

	public EuclidDistance(int size) {
		super(size);
	}

	@Override
	public double evaluate(Data d1, Data d2) {
		double len = 0;
		for(int i = 0; i<d1.featureValues.length; i++){
			len += (Math.pow((d1.getFeatureValueAtIndex(i) - d2.getFeatureValueAtIndex(i)), 2));
		}
		len = Math.sqrt(len);
		return -len;
	}

}

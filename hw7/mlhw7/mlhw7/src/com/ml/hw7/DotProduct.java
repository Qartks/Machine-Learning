package com.ml.hw7;

public class DotProduct extends Kernel {

	public DotProduct(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}

	@Override
	public double evaluate(Data x1, Data x2) throws Exception {
		double dotProduct = 0;
		int featureSize = x1.getFeatureValues().length;
		for(int i=0; i < featureSize; i++) {
			dotProduct+= x1.getFeatureValueAtIndex(i) * x2.getFeatureValueAtIndex(i);
		}
		return dotProduct;
	}

}

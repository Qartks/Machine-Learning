package com.ml.hw7;


public class GaussianKernel extends Kernel {
	
	double nu;

	public GaussianKernel(int size, double nu) {
		super(size);
		this.nu = nu;
	}

	@Override
	public double evaluate(Data x1, Data x2) throws Exception {
		int featureSize = x1.getFeatureValues().length;
		double vectorDiff = 0;
		for(int i=0; i < featureSize; i++) {
			vectorDiff+= Math.pow((x1.getFeatureValueAtIndex(i) - x2.getFeatureValueAtIndex(i)),2);
		}
		vectorDiff/= (2*nu*nu);
		return Math.exp(-vectorDiff);
	}

}

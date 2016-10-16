package com.ml.hw7;

public class PolynomialKernel extends Kernel {

	double a;
	double b;
	double d;
	
	public PolynomialKernel(int size, double a, double b, double d) {
		super(size);
		this.a = a;
		this.b = b;
		this.d = d;
	}

	@Override
	public double evaluate(Data x1, Data x2) throws Exception {
		double dotProduct = calculateDotProduct(x1, x2);
		double result = a*dotProduct;
		result+= b;
		return Math.pow(result, d);
	}
	
	private double calculateDotProduct(Data x1, Data x2) {
		double dotProduct = 0;
		int featureSize = x1.getFeatureValues().length;
		for(int i=0; i < featureSize; i++) {
			dotProduct+= x1.getFeatureValueAtIndex(i) * x2.getFeatureValueAtIndex(i);
		}
		return dotProduct;
	}

}

package com.ml.hw7;

public class CosineDistance extends Kernel {

	public CosineDistance(int size) {
		super(size);
	}

	@Override
	public double evaluate(Data x1, Data x2) throws Exception {
		double dotProduct = calculateDotProduct(x1, x2);
		double normX1 = calculateNormOfData(x1);
		double normX2 = calculateNormOfData(x2);
		double similarity = dotProduct/(normX1 * normX2);
		return similarity;
	}

	private double calculateDotProduct(Data x1, Data x2) {
		double dotProduct = 0;
		int featureSize = x1.getFeatureValues().length;
		for(int i=0; i < featureSize; i++) {
			dotProduct+= x1.getFeatureValueAtIndex(i) * x2.getFeatureValueAtIndex(i);
		}
		return dotProduct;
	}
	
	private static double calculateNormOfData(Data x1) throws Exception {
		double norm = 0;
		int featureSize = x1.getFeatureValues().length;
		for(int i=0; i < featureSize; i++) {
			norm+= Math.pow(x1.getFeatureValueAtIndex(i),2);
		}
		return Math.sqrt(norm);
	}

}

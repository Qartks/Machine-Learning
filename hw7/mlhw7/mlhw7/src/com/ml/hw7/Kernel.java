package com.ml.hw7;

public abstract class Kernel {
	
	private double[][] cache;
	
	public Kernel(int size) {
		cache = new double[size][size];
		for(int i=0;i<size;i++) {
			for(int j=0;j<size;j++) {
				cache[i][j] = Double.NaN;
			}
		}
	}
	
	public double computeValue(Data x1, Data x2, int indexX1, int indexX2, boolean useCache) throws Exception{
		double value = Double.NaN;
		if(useCache) {
			value = cache[indexX1][indexX2];
			if(Double.isNaN(value)) {
				value = evaluate(x1, x2);
				cache[indexX1][indexX2] = value;
				cache[indexX2][indexX1] = value;
			}
		} else {
			value = evaluate(x1, x2);
		}
		return value;
	}

	public abstract double evaluate(Data x1, Data x2) throws Exception;

}

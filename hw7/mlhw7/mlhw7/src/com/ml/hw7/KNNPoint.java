package com.ml.hw7;

public class KNNPoint {
	
	double similarity;
	Data p;
	
	public KNNPoint(Data di, double k) {
		this.p = di;
		this.similarity = k;
	}
	
	public double getLabel(){
		return p.labelValue;
	}
	
	
}

package com.ml.hw5;

import org.ejml.simple.SimpleMatrix;

public class GaussianModel {

	// Mean => 1xM - Mean for each model
	// CoVariance Matrix => MxM
	
	SimpleMatrix µ;
	SimpleMatrix ¥;
	
	public GaussianModel() {
		
	}

	public SimpleMatrix getΜean() {
		return µ;
	}

	public void setMean(SimpleMatrix µ) {
		this.µ = µ;
	}

	public SimpleMatrix getCovariance() {
		return ¥;
	}

	public void setCovariance(SimpleMatrix ¥) {
		this.¥ = ¥;
	}
	
	public String toString(){
		
		StringBuilder sb = new StringBuilder();
		sb.append("\n");
		sb.append(µ.toString());
		sb.append("\n");
		sb.append(¥.toString());
		sb.append("\n");
		
		return sb.toString().trim();
	}
}

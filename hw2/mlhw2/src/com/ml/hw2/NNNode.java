package com.ml.hw2;

public class NNNode {
	
	private double netInput;
	private double output;
	private double targetOutput;
	
	public NNNode() {
		this.netInput = 0;
		this.output = 0;
		this.targetOutput = 0;
	}
	
	public NNNode(double netInput, double output, double targetOutput) {
		this.netInput = netInput;
		this.output = output;
		this.targetOutput = targetOutput;
	}

	public double getNetInput() {
		return netInput;
	}
	
	public void setNetInput(double netInput) {
		this.netInput = netInput;
		this.calculateOutput();
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

	public void calculateOutput() {
		this.setOutput(getLogisticRegressionOutput());
	}
	
	private double getLogisticRegressionOutput() {
		return 1/(1+Math.exp(-netInput));
	}

	public double gettargetOutput() {
		return targetOutput;
	}

	public void settargetOutput(double targetOutput) {
		this.targetOutput = targetOutput;
	}
	
	public double getGradiant() {
		return output * (1-output);
	}
	
	public double getError() {
		return targetOutput - output;
	}

}

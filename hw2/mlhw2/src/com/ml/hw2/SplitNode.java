package com.ml.hw2;

public class SplitNode {
	
	private int splitFeatureIndex;
	private double splitFeatureValue;
	private double infoGain;
	private TreeNode noBranch;
	private TreeNode yesBranch;
	
	public int getSplitFeatureIndex() {
		return splitFeatureIndex;
	}
	public void setSplitFeatureIndex(int splitFeatureIndex) {
		this.splitFeatureIndex = splitFeatureIndex;
	}
	public double getSplitFeatureValue() {
		return splitFeatureValue;
	}
	public void setSplitFeatureValue(double splitFeatureValue) {
		this.splitFeatureValue = splitFeatureValue;
	}
	public double getInfoGain() {
		return infoGain;
	}
	public void setInfoGain(double infoGain) {
		this.infoGain = infoGain;
	}
	public TreeNode getNoBranch() {
		return noBranch;
	}
	public void setNoBranch(TreeNode noBranch) {
		this.noBranch = noBranch;
	}
	public TreeNode getYesBranch() {
		return yesBranch;
	}
	public void setYesBranch(TreeNode yesBranch) {
		this.yesBranch = yesBranch;
	}

}

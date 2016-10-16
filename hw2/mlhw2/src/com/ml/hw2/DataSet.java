package com.ml.hw2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class DataSet {
	
	private List<Data> data;
	private List<Feature> features;
	private int labelIndex;
	private HashSet<Integer> classes;
	private int classNum;

	public DataSet(int labelIndex, List<Feature> features) {
		this.labelIndex = labelIndex;
		this.features = features;
		data = new ArrayList<Data>();
		classes = new HashSet<Integer>();
	}
	
	public int dataSize(){
		if(data == null) {
			return 0;
		}
		return data.size();
	}
	
	public List<Feature> getFeatures() throws Exception {
		if(features == null || features.size() == 0) {
			throw new Exception("Feature is null or empty");
		}
		return features;
	}
	
	public Feature getFeature(int index) throws Exception {
		if(features == null || features.size() == 0) {
			throw new Exception("The Feature Set is Empty");
		}
		if( (index < 0) || (index > (features.size() -1))) {
			throw new Exception("Incorrect Index");
		}
		return features.get(index);
	}
	
	public void addData(Data data) throws Exception {
		data.setDataSet(this);
		this.data.add(data);
		if(features.get(labelIndex).isNominal()) {
			Integer classLabel = new Integer((int) data.labelValue());
			classes.add(classLabel);
			classNum = classes.size();
		}
	}
	
	public int getLabelIndex() {
		return this.labelIndex;
	}
	
	public void setClassNum(int classNum) {
		this.classNum = classNum;
	}
	
	public int classNum() {
		return classNum;
	}
	
	public boolean isClassificationTask() {
		return features.get(labelIndex).isNominal();
	}
	
	public List<Data> getData() {
		return data;
	}
	
	@Override
	public String toString() {
		StringBuilder stringBuilder = new StringBuilder();
		for(Data d : data) {
			stringBuilder.append(d.toString()+"\n");
		}
		return stringBuilder.toString();
	}
	
	public double[] getLeftData(int i, int featureIndex) throws Exception{
		double[] left = new double[i+1];
		for(int x= 0; x< i+1; x++){
			left[x] = this.getData().get(x).labelValue();
		}
		return left;
	}
	
	public double[] getRightData(int i, int featureIndex) throws Exception{
//		System.out.println(this.getData().size());
		double[] right = new double[this.getData().size() - (i+1)];
		for(int x= 0; x< this.getData().size() - (i+1); x++){
			right[x] = this.getData().get(x + (i+1)).labelValue();
		}
		return right;
	}

	public double[] getFeatureMatrix() throws Exception {

		double featureSize = this.getFeatures().size();
		double dataSize = this.getData().size();
		
		double[] featureMatrix = new double[(int) (featureSize * dataSize)];
		int i = 0;
		
		for(Data d : this.getData()){
			featureMatrix[i] = 1;
			i++;
			for(int j = 0; j <featureSize - 1; j++){
				featureMatrix[i] = d.getFeatureValue(j);
				i++;
			}
		}
		//		
//		for(int i = 0; i< this.getData().size(); i++){
//			Data d = this.getData().get(i);
//			for(int j = 0; j < this.getFeatures().size() - 1; j++){
//				featureMatrix[(this.getFeatures().size() - 1) * i + j] = d.getFeatureValue(j);
//			}
//		}

		return featureMatrix;
	}

	public double[] getLabelMatrix() throws Exception {
		
		double[] labelMatrix = new double[this.dataSize()];
		
		for(int i = 0; i< this.getData().size(); i++){
			labelMatrix[i] = this.getData().get(i).labelValue();
		}
		
		return labelMatrix;
	}
}
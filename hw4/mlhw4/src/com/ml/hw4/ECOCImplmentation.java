package com.ml.hw4;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ECOCImplmentation {
	
	int noOfClassifiers;
	int noOfLabels;
	double[][] errTable =  {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
							{1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0},
							{1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0},
							{0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0},
							{0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},
							{1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0},
							{1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1},
							{0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1}};
	
	List<AdaBoostImplementation> classifiers = new ArrayList<AdaBoostImplementation>();
	
	public ECOCImplmentation(int n, int x) {
		this.noOfClassifiers = n;
		this.noOfLabels = x;
		
//		this.errTable = new double[x][n];
	}

	public void generateErrorTable() {
		List<String> list = new ArrayList<String>();
		while(list.size() != noOfClassifiers){
			int num = (int) (Math.random() * Math.pow(2, noOfLabels));
			String s = padLeft(Integer.toBinaryString(num), 8).replace(' ', '0');
			String s2 = flipBits(s);
			int count = getCount(s,'1');
			if(count < 3){
				continue;
			}
			if(!list.contains(s) && !list.contains(s2)){
				list.add(s);
			}
		}
		
		for(int i = 0; i < noOfClassifiers; i++){
			for(int j = 0; j < noOfLabels; j++){
				errTable[j][i] = list.get(i).charAt(j) - 48;
//				if(errTable[j][i] == 0){
//					errTable[j][i] = -1;
//				}
			}
		}
		
		for(int i = 0; i < noOfLabels; i++){
			for(int j = 0; j < noOfClassifiers; j++){
				System.out.print((int)errTable[i][j] + " ");
			}
			System.out.println();
		}
		
	}
	
	private String flipBits(String s) {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < s.length(); i++){
			if(s.charAt(i) == '0'){
				sb.append('1');
			} else {
				sb.append('0');
			}
		}
		return sb.toString().trim();
	}

	private int getCount(String s, char c) {
		int counter = 0;
	    for (int i = 0; i < s.length(); i++) {
	      if (s.charAt(i) == c)
	        counter++;
	    }

		return counter;
	}
	
	public static String padLeft(String s, int n) {
	    return String.format("%1$" + n + "s", s);  
	}
	
	public void train(DataSet trainData, DataSet testData) throws Exception {
//		generateErrorTable();
		
		int adaClassifiers = 2000;
		boolean optimal = false;
		List<Double> lab = new ArrayList<Double>();
		
		for(Data d : trainData.getData()){
			lab.add(d.getLabelValue());
		}
		
		for(int i = 0; i < noOfClassifiers; i++){
			
			System.out.println("Classifier " + (i+1) + ":");
			int count = 0;
			for(Data d : trainData.getData()){
				d.setLabelValue(lab.get(count));
				d.setLabelValue(newLabelValue(i,d));
				count++;
			}
			
			AdaBoostImplementation ada = new AdaBoostImplementation(trainData, adaClassifiers);
			ada.train(trainData, null , adaClassifiers, optimal);
			classifiers.add(ada);
			
//			for(Data d : testData.getData()){
//				System.out.print(ada.predictValue(d) + " " + d.labelValue);
//				System.out.println("");
//			}
		}
		
		System.out.println("hello");
	}

	private double newLabelValue(int i, Data d) {
		double l = d.getLabelValue();
		return errTable[(int) l][i];
	}

	public void test(DataSet testData) {
		
		double error = 0;
		for(Data d : testData.getData()){
			double[] res = new double[noOfClassifiers];
			int count = 0;
			for(AdaBoostImplementation ada : classifiers){
//				System.out.print(ada.predictValue(d) + " " + d.labelValue);
				res[count] = ada.predictValue(d);
				count++;
			}
			int predLabel = getLabelHammingDist(res);
			int actualLabel = (int) d.getLabelValue();
//			System.out.println(Arrays.toString(res) + " -> " + predLabel + actualLabel);
			if(predLabel != actualLabel){
				error++;
			}
			
//			System.out.println("");
		}
		
		System.out.println("Accuracy :" + (1 - (error/testData.getDataSize())));
	}

	private int getLabelHammingDist(double[] res) {
		int label = 0;
		int minDist = Integer.MAX_VALUE;
		
		for(int i = 0 ; i < noOfLabels; i++){
			int hamDist = 0;
			
			for(int j = 0; j < noOfClassifiers; j++){
				if(errTable[i][j] != res[j]){
					hamDist++;
				}
			}
			
			if(hamDist < minDist){
				minDist = hamDist;
				label = i;
			}
		}
		return label;
	}
	

}

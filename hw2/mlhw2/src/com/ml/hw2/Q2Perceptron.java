package com.ml.hw2;

import java.io.BufferedReader;
import java.io.FileReader;

import org.ejml.simple.SimpleMatrix;

public class Q2Perceptron {

	public static void main(String[] args) throws Exception {

		BufferedReader readFile = null;
		String fileName = "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW2/perceptronData.txt";
		
		int noOfData = 1000;
		int noOFFeatures = 5;
		
		double[] featureList = new double[noOFFeatures * noOfData];
		double[] labelList = new double[noOfData];
		double[] weights = new double[noOFFeatures];
		int i = 0;
		int k = 0;
		try {
			
			readFile = new BufferedReader(new FileReader(fileName));
			
			while(true){
				String line = readFile.readLine();
				if (line == null) {
					break;
				}

				if (line.trim().length() == 0) {
					break;
				}
				
				String delims = "\\s+|,";
				String[] values = line.trim().split(delims);
				
//				System.out.println(Arrays.toString(values));
				
				featureList[i] = 1;
				i++;
				for(int j = 0; j < values.length - 1; j++){
					featureList[i] = Double.parseDouble(values[j]);
					i++;
				}
				labelList[k++] = Double.parseDouble(values[values.length -1]);
				
			}
			
		} finally {
			
			readFile.close();
			
		}
		
		for(int i1 = 0; i1< noOFFeatures; i1++){
			weights[i1] = Math.random();
		}
		
		SimpleMatrix x = new SimpleMatrix(noOfData, noOFFeatures, true, featureList);
		SimpleMatrix y = new SimpleMatrix(noOfData, 1, true, labelList);
		SimpleMatrix w = new SimpleMatrix(noOFFeatures, 1, true, weights);
		
		double lambda = 0.001;
		
		PerceptronImplementation.learn(w, x, y, lambda);
		
	}

}

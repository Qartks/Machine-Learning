package com.ml.hw4;

public class Q7Main {
	
	public static void main(String[] args) throws Exception {
		
		DataSet2 trainingData = OldDataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt",  "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, true);
		DataSet2 testData = OldDataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_test.txt", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, false);
		
		
		GradBoostImplementation gb = new GradBoostImplementation();
		
		gb.boooostAndTrain(10, trainingData, testData);
		
	}
	

}

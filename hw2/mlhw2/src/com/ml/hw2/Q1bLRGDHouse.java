package com.ml.hw2;


public class Q1bLRGDHouse {

	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt", 
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt");
		
		DataSet testSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_test.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt");

		
		Normalize.normalizeDataSandS(dataSet);
		Normalize.normalizeDataSandS(testSet);
//		Normalize.normalizeData(dataSet);
		
		double lambda = 0.0005;
		double threshold = 0.001;
//		GDImplementation.learn(dataSet, lambda, false, threshold);
		

		GradDesImplement.learn(dataSet, lambda, false, threshold, false, testSet);
		
	}

}

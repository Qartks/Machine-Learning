package com.ml.hw5;

public class Q3MainA {

	public static void main(String[] args) throws Exception {
		DataSet trainData = DataInputer.getPollutedData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/train_feature.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/train_label.txt", 1057);
		DataSet testData = DataInputer.getPollutedData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/test_feature.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/test_label.txt", 1057);
		
		Normalize.normalizeDataSandS(trainData, testData);
		
		GradDesImplement.learn(trainData, 0.001, true, 0, true, testData);
		
	}

}

package com.ml.hw5;

public class Q2MainB {

	public static void main(String[] args) throws Exception {
		DataSet trainData = DataInputer.getPollutedData("/Users/kartikeyashukla/Documents/MATLAB/trainPCA.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/train_label.txt", 100);
		DataSet testData = DataInputer.getPollutedData("/Users/kartikeyashukla/Documents/MATLAB/testPCA.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/test_label.txt", 100);
		
		Normalize.normalizeDataSandS(trainData, testData);
		
		NaiveBayesGaussian nbg = new NaiveBayesGaussian();
		
		GaussParams gp = nbg.train(trainData);
		ErrorStat errTest = nbg.test(gp, testData, 1);
		
		System.out.println("Gaussian -> " + errTest.getAccuracy());
		System.out.println(errTest);
	}

}

package com.ml.hw5;

public class Q2MainA {

	public static void main(String[] args) throws Exception {
		DataSet trainData = DataInputer.getPollutedData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/train_feature.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/train_label.txt", 1057);
		DataSet testData = DataInputer.getPollutedData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/test_feature.txt",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/test_label.txt", 1057);
		

		Normalize.normalizeDataSandS(trainData, testData);
		
		NaiveBayesGaussian nbg = new NaiveBayesGaussian();
		
		GaussParams gp = nbg.train(trainData);
		ErrorStat errTest = nbg.test(gp, testData, 1);
		
//		System.out.println(nbg.rocData.toString());
		System.out.println("Gaussian -> " + errTest.getAccuracy());
	}

}

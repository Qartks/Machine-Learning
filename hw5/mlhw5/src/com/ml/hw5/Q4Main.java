package com.ml.hw5;

public class Q4Main {

	public static void main(String[] args) throws Exception {
		
		DataSet trainData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/20_percent_missing_train.txt");
		DataSet testData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/20_percent_missing_test.txt");
		
		NaiveBayesBernoulli nbb = new NaiveBayesBernoulli();
		
		NBBStats nb = nbb.train(trainData);
		ErrorStat err = nbb.test(nb, testData, 0);
		
		System.out.println(err.getAccuracy());
		
	}

}

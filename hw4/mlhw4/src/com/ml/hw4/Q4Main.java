package com.ml.hw4;


public class Q4Main {

	public static void main(String[] args) throws Exception {
		
		DataSet trainData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/train.trec/feature_matrix.txt");
		DataSet testData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/test.trec/feature_matrix.txt");
		
		ECOCImplmentation ecoc = new ECOCImplmentation(20, 8);
		
		trainData.computeFeatureStats();
		ecoc.train(trainData, testData);
		
		trainData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/train.trec/feature_matrix.txt");
		
		ecoc.test(trainData);
		ecoc.test(testData);
		
		System.out.println("Hi");
	}

}

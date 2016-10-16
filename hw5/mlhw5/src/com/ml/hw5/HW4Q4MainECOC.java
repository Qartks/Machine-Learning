package com.ml.hw5;

import java.util.Collections;


public class HW4Q4MainECOC {

	public static void main(String[] args) throws Exception {
		
		DataSet trainData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/train.trec/feature_matrix.txt");
		DataSet testData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/test.trec/feature_matrix.txt");
		
		ECOCImplmentation ecoc = new ECOCImplmentation(30, 8);
		
		Collections.shuffle(trainData.getData());
		Collections.shuffle(trainData.getData());
		
		int size = trainData.getDataSize();
		int c = (int) (0.40 * size);
		
		DataSet T = new DataSet(trainData.getFeatureSize());
		DataSet X = new DataSet(trainData.getFeatureSize());
		for(int i = 0; i< size; i++){
			Data d = trainData.getData().get(i);
			if( i < c){				
				T.addData(d);
			} else {
				X.addData(d);
			}
		}
		
		T.computeFeatureStats();
		trainData.computeFeatureStats();
		ecoc.train(T, testData, 2000, false, false);
		
		
		trainData = DataInputer.getECOCData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW4/8newsgroup/train.trec/feature_matrix.txt");
		
		System.out.print("Train ");
		ecoc.test(trainData);
		System.out.print("Test ");
		ecoc.test(testData);
		
		System.out.println("Hi");
	}

}

package com.ml.hw3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Q2MainBucket9 {

	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		allData.computeFeatureStats();
		
		errorWithKFold(allData, 10, 9);
//		getROCData(allData, 9);
		
		
	}
	
	private static void getROCData(DataSet dataSet, int N) {

		int dataPerFold = dataSet.getDataSize()/ 10;
		
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold< 10 ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			trainingData.setFeatureStat(dataSet.getFeatureStat());
			testData.setFeatureStat(dataSet.getFeatureStat());
			
			NaiveBayesBucket9 n9 = new NaiveBayesBucket9(N, dataSet.getFeatureSize());
			List<Double> threshList = null;
			
			for(int x = 0; x < dataSet.getDataSize(); x++) {
				Data d = dataSet.getData().get(x);
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(d);
					d.setDataSet(testData);
				} else {
					trainingData.addData(d);
					d.setDataSet(trainingData);
				}
			}
			
			Bucket9 b9 = n9.train(trainingData);
			ErrorStat err = n9.test(b9, testData, 0.5);
			
			threshList =  new ArrayList<Double>(n9.rocData);
			Collections.sort(threshList);
			
			for(Double thres: threshList){
				ErrorStat errTest2 = n9.test(b9, testData, thres);
				System.out.println(errTest2.getFPRate() + "\t" + errTest2.getTPRate());
			}

			if(fold == 0)
				break;
		}
	}
	
	
	private static void errorWithKFold(DataSet dataSet, int k, int N) {
		
		double avgTestError = 0;
		double avgTrainError = 0;
		ErrorStat avgError = new ErrorStat();
		int dataPerFold = dataSet.getDataSize()/ k;
		
		Collections.shuffle(dataSet.getData());
		for(int fold=0; fold< k ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			trainingData.setFeatureStat(dataSet.getFeatureStat());
			testData.setFeatureStat(dataSet.getFeatureStat());
			
			NaiveBayesBucket9 n9 = new NaiveBayesBucket9(N, dataSet.getFeatureSize());
			
			for(int x = 0; x < dataSet.getDataSize(); x++) {
				Data d = dataSet.getData().get(x);
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(d);
					d.setDataSet(testData);
				} else {
					trainingData.addData(d);
					d.setDataSet(trainingData);
				}
			}
			Bucket9 b9 = n9.train(trainingData);
			ErrorStat err = n9.test(b9, trainingData, 0.5);
			avgTrainError += err.getAccuracy();
			System.out.println("Training Error For Fold " + (fold+1) + " -> " + err.getAccuracy());
			
			ErrorStat errTest = n9.test(b9, testData, 0.5);
			avgTestError += errTest.getAccuracy();
			System.out.println("Testing Error For Fold " + (fold+1) + " -> " + errTest.getAccuracy());
			System.out.println("Prob Error Rate -> " + errTest.toString());
			System.out.println();
			avgError.incrementErrorRates(errTest.errR);
		}
		
		for(int i = 0; i< 4; i++){
			avgError.errR[i] /= k;
		}
		
		System.out.println();
		System.out.println("Average Training Error -> " + avgTrainError/k);
		System.out.println("Average Testing Error -> " + avgTestError/k);
		System.out.println("Average Error Rate" + avgError.toString());
		
	}

}

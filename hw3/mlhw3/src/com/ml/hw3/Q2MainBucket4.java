package com.ml.hw3;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Q2MainBucket4 {

	public static void main(String[] args) throws Exception {

		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW3/spambase.data");
		allData.computeFeatureStats();
		
		errorWithKFold(allData, 10);
//		getROCData(allData);
		
	}
	
	private static void getROCData(DataSet dataSet) {

		int dataPerFold = dataSet.getDataSize()/ 10;
		
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold< 10 ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			trainingData.setFeatureStat(dataSet.getFeatureStat());
			testData.setFeatureStat(dataSet.getFeatureStat());
			
			NavieBayesBucket4 nb4 = new NavieBayesBucket4();
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
			
			Bucket4 b4 = nb4.train(trainingData);
			ErrorStat err = nb4.test(b4, testData, 0.5);
			
			threshList =  new ArrayList<Double>(nb4.rocData);
			Collections.sort(threshList);
			
			for(Double thres: threshList){
				ErrorStat errTest2 = nb4.test(b4, testData, thres);
				System.out.println(errTest2.getFPRate() + "\t" + errTest2.getTPRate());
			}

			if(fold == 0)
				break;
		}
	}
	
	
	private static void errorWithKFold(DataSet dataSet, int k) {
		
		double avgTestError = 0;
		double avgTrainError = 0;
		ErrorStat avgError = new ErrorStat();
		int dataPerFold = dataSet.getDataSize()/ k;
		
		
		for(int fold=0; fold< k ; fold++) {
			
			Collections.shuffle(dataSet.getData());
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			trainingData.setFeatureStat(dataSet.getFeatureStat());
			testData.setFeatureStat(dataSet.getFeatureStat());
			
			NavieBayesBucket4 nb4 = new NavieBayesBucket4();
			
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
			Bucket4 b4 = nb4.train(trainingData);
			ErrorStat err = nb4.test(b4, trainingData, 0.5);
			avgTrainError += err.getAccuracy();
			System.out.println("Training Error For Fold " + (fold+1) + " -> " + err.getAccuracy());
			
			ErrorStat errTest = nb4.test(b4, testData, 0.5);
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

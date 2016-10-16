package com.ml.hw5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Q5Main {
	
	public static void main(String[] args) throws Exception{
		// TODO Auto-generated method stub
		
		MNISTReader trainData = new MNISTReader();
		MNISTReader testData = new MNISTReader();
		
		trainData.ReadImages("train-labels-idx1-ubyte", "train-images-idx3-ubyte");
		testData.ReadImages("t10k-labels-idx1-ubyte", "t10k-images-idx3-ubyte");
		
		// Load Images 
		// Generate Random Rectangles for a 28x28 image frame -> Store in an Map
		// For each Image:
		// 		Generate Map with count of black points with origin as the first point
		// 		Make two classifiers for HAAR features
		//			Count Black in rectangle with help of origin table using DP
		//			Calculate horizontal/vertical feature -> 
		// 
		
		List<Image> trainImgs = trainData.getImgList();
		List<Image> testImgs = testData.getImgList();
		
		List<Image> sampledTrainImageList = getSampledImageList(0.20, trainImgs);
		
		ImageFeatureExtraction ife = new ImageFeatureExtraction();
		
		ife.generateCache(sampledTrainImageList);
		ife.generateRandomRectangles(100);
		DataSet train = ife.getHAARFeatureData(sampledTrainImageList);
		
		System.out.println("Training Data Processing Complete\n");
		
		ImageFeatureExtraction ifeTest = new ImageFeatureExtraction();
		
		ifeTest.generateCache(testImgs);
		ifeTest.calculateRectangleDataForTest(ife.testRectangles);
		DataSet test = ifeTest.getHAARFeatureData(testImgs);
		
		System.out.println("Test Data Processing Complete\n");
		
//		Normalize.normalizeDataSandS(train, test);
//		System.out.println("Normalized\n");
		
		ECOCImplmentation ecoc = new ECOCImplmentation(50, 10);
		
		System.out.println("Starting ECOC...\n");
		ecoc.train(train, test, 2000, false, false);
		ecoc.test(test);
		
		
	}

	private static List<Image> getSampledImageList(double d, List<Image> trainImgs) {
		
		int[] count = new int[10];
		for(Image img : trainImgs){
			count[(int)img.getLabel()]++;
		}
		
		for(int i = 0; i <=9; i++){
			count[i] *= d;
		}
		
		Collections.shuffle(trainImgs);
		
		List<Image> sampledTrainImageList = new ArrayList<Image>();
		for(Image img : trainImgs){
			int label = (int)img.getLabel();
			if(count[label] > 0){
				sampledTrainImageList.add(img);
				count[label]--;
			}
		}
		return sampledTrainImageList;
	}

}

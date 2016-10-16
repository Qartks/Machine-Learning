package com.ml.hw7;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


public class Q1MainB {
	
	public static void main (String args[]) throws Exception{
//		generateFiles(0.10, 0.9);
		
		executeDigitsKNN();
		
		
	}
	
	private static void executeDigitsKNN() throws Exception {
		
		DataSet trainData = DataInputer.getData("C:/Users/eesha_000/Downloads/ML/Q3_image_train.txt");
		DataSet testData = DataInputer.getData("C:/Users/eesha_000/Downloads/ML/Q3_image_test.txt");
		int size = trainData.getDataSize();
		
		Normalize.normalizeDataSandS(trainData, testData);
		
//		Kernel kernel = new CosineDistance(size);
		Kernel kernel = new PolynomialKernel(size, 0.1, 1, 2);
//		Kernel kernel = new GaussianKernel(size, 0.5);
		KNNImplementation knn = new KNNImplementation(1, 0, false, trainData, kernel);
		double error = knn.execute(testData);
		System.out.println("Accuracy -> " + (1 - error) * 100);
		
	}

	private static void generateFiles(double trainPerct, double testPerct) throws Exception {
		
		
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
		
		List<Image> sampledTrainImageList = getSampledImageList(trainPerct, trainImgs);
		
		ImageFeatureExtraction ife = new ImageFeatureExtraction();
		
		ife.generateCache(sampledTrainImageList);
		ife.generateRandomRectangles(100);
		DataSet train = ife.getHAARFeatureData(sampledTrainImageList);
		
		System.out.println("Training Data Processing Complete\n");
		
		List<Image> sampledTestImageList = getSampledImageList(testPerct, testImgs);
		
		ImageFeatureExtraction ifeTest = new ImageFeatureExtraction();
		
		ifeTest.generateCache(testImgs);
		ifeTest.calculateRectangleDataForTest(ife.testRectangles);
		DataSet test = ifeTest.getHAARFeatureData(testImgs);
		
		System.out.println("Test Data Processing Complete\n");
		
		Normalize.normalizeDataSandS(train, test);

		DataInputer.writeDataToFile(
				"C:/Users/eesha_000/Downloads/ML/Q3_image_train.txt", train);
		DataInputer.writeDataToFile(
				"C:/Users/eesha_000/Downloads/ML/Q3_image_test.txt", test);
		
		for(int i = 0; i < 10; i++){
			DataSet label = new DataSet(200);
//			Collections.shuffle(train.getData());
			for(int j = 0; j< train.getDataSize(); j++){
				Data d = train.getData().get(j);
				label.addData(d);
			}
			DataInputer.writeDataToFile(
					"C:/Users/eesha_000/Downloads/ML/image_train_label_"+i+".txt", label);
		}
		System.out.println("\n\nFiles Written !\n\n");
		
		
	}

	private static List<Image> getSampledImageList(double d, List<Image> trainImgs) {
		
		DataSet[] ds = new DataSet[10];
		
		for(int i = 0 ; i < 10 ; i++){
			ds[i] = new DataSet(200);
		}
		
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

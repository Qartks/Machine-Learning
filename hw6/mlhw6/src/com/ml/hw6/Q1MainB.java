package com.ml.hw6;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


import libsvm.*;
import net.sf.javaml.classification.Classifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

public class Q1MainB {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
//		generateFiles();
		applyLibSVM();
		
	}
	
	private static void applyLibSVM() throws Exception {
		Dataset data = FileHandler.loadDataset(new File("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_train.txt"), 200, " ");
		
		Classifier svm = new LibSVM();
        svm.buildClassifier(data);
        
        Dataset dataForClassification = FileHandler.loadDataset(new File("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_test.txt"), 200, " ");
        float correct = 0, wrong = 0;
        
        for (Instance inst : dataForClassification) {
            Object predictedClassValue = svm.classify(inst);
            Object realClassValue = inst.classValue();
            if (predictedClassValue.equals(realClassValue))
                correct++;
            else
                wrong++;
        }
        
        System.out.println("Correct predictions  " + correct);
        System.out.println("Wrong predictions " + wrong);
        System.out.println("Accuracy: " + (correct / (correct + wrong)));
        
	}

	private static void generateFiles() throws Exception {
		

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
		
		Normalize.normalizeDataSandS(train, test);

		DataInputer.writeDataToFile(
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_train.txt", train);
		DataInputer.writeDataToFile(
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_test.txt", test);
		
		System.out.println("\n\nFiles Written !\n\n");
		
		
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

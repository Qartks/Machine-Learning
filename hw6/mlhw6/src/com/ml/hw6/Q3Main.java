package com.ml.hw6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Q3Main {
	
	static List<Double> labelList = new ArrayList<Double>();

	public static void main(String[] args) throws Exception {
		
		DataInputer.noOFFeatures=200;
//		generateFiles(0.05, 0.6);
		executeDigitSet();
		
	}
	

	private static void executeDigitSet() throws Exception {
		SMO[] smos = new SMO[10];
		
		DataSet trainData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/Q3_image_train.txt");
		DataSet testData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/Q3_image_test.txt");
		
		List<Double > trainList = new ArrayList<Double>();
		for(int j = 0; j< trainData.getDataSize(); j++){
			Data d = trainData.getData().get(j);
			trainList.add(d.labelValue);
		}
		
		List<Double > testList = new ArrayList<Double>();
		for(int j = 0; j< testData.getDataSize(); j++){
			Data d = testData.getData().get(j);
			testList.add(d.labelValue);
		}
		
		for(int i = 0; i < smos.length; i++){
			System.out.println("Training " + (i) + " SMO");
			DataSet labelData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_train_label_"+i+".txt");
			SMO smo = new SMO(1, 0.001, 0.01, labelData, 20);
			smo.setTrainingLabels((double)i,trainList);
			smo.buildLagrangeMultipliers();
			smos[i] = smo;
		}
		getAccuracy(trainData, smos, trainList);
		getAccuracy(testData, smos, testList);
		
	}


	private static void getAccuracy(DataSet dataSet, SMO[] smos, List<Double> list) {
		double error = 0;
		for(Data d : dataSet.getData()){
			double actualValue = d.getLabelValue();
			double predictedValue = getPredictedValue(smos, d, list);
			if(predictedValue != actualValue){
				error++;
			}
		}
		System.out.println("Accuracy : " + ((1 - error / dataSet.getDataSize())*100));
	}


	private static double getPredictedValue(SMO[] smos, Data d, List<Double> list) {
		double maxFx = Double.NEGATIVE_INFINITY;
		double maxLabel = 0;
		for(int i = 0; i < smos.length; i++){
			SMO smo = smos[i];
//			smo.setTrainingLabels((double)i, list);
			double fxVal = smo.getFXvalue(d);
			if(fxVal > maxFx){
				maxFx = fxVal;
				maxLabel = (double) i;
			}
		}
		return maxLabel;
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
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/Q3_image_train.txt", train);
		DataInputer.writeDataToFile(
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/Q3_image_test.txt", test);
		
		labelList = new ArrayList<Double>();
		for(int j = 0; j< train.getDataSize(); j++){
			Data d = train.getData().get(j);
			labelList.add(d.labelValue);
		}
		System.out.println(labelList.toString());
		for(int i = 0; i < 10; i++){
			DataSet label = new DataSet(200);
//			Collections.shuffle(train.getData());
			for(int j = 0; j< train.getDataSize(); j++){
				Data d = train.getData().get(j);
				label.addData(d);
			}
			DataInputer.writeDataToFile(
					"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/image_train_label_"+i+".txt", label);
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

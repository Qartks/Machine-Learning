package com.ml.hw6;


public class Normalize {
//	
	static void normalizeDataSandS(DataSet trainSet) throws Exception {

		int noOfFeatures = trainSet.getFeatureSize();
		double[] featureMin = new double[noOfFeatures];
		double[] featureMax = new double[noOfFeatures];
		
		for(int i = 0; i<noOfFeatures; i++){
			featureMin[i] = Double.POSITIVE_INFINITY;
			featureMax[i] = Double.NEGATIVE_INFINITY;
		}
		
		for(int i = 0; i< noOfFeatures; i++){
			
			for(Data d: trainSet.getData()){
				
				if(d.getFeatureValueAtIndex(i) > featureMax[i]){
					featureMax[i] = d.getFeatureValueAtIndex(i);
				}
				if(d.getFeatureValueAtIndex(i) < featureMin[i]){
					featureMin[i] = d.getFeatureValueAtIndex(i);
				}
				
			}
			
		}
		
		for(Data d : trainSet.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				featureValues[i] = ((featureValues[i] - featureMin[i]) / (featureMax[i] - featureMin[i]));
			}
		}
		
	}
//	
//// static void normalizeData(DataSet trainSet) throws Exception {
////		
////		int noOfFeatures = trainSet.getFeatures().size();
////		double[] featureMean = new double[noOfFeatures];
////		
////		int trainDataSize = trainSet.getData().size();
////		
////		for(int i = 0; i< noOfFeatures; i++){
////			
////			for(Data d: trainSet.getData()){
////				featureMean[i] += d.getFeatureValue(i); 
////			}
////			
////		}
////		
////		for(int i = 0; i< noOfFeatures; i++){
////			featureMean[i] /= featureMean[i]/(trainDataSize);
////		}
////		
////		double[] featureVar = new double[noOfFeatures];
////		
////		for(int i = 0; i< noOfFeatures; i++){
////			
////			for(Data d: trainSet.getData()){
////				featureVar[i] += Math.pow(d.getFeatureValue(i) - featureMean[i],2);
////			}
////			
////		}
////		
////		for(int i = 0; i< featureVar.length; i++){
////			featureVar[i] = Math.sqrt(featureVar[i]);
////		}
////		
////		for(Data d : trainSet.getData()){
////			double[] featureValues = d.getFeatureValues();
////			
////			for(int i = 0; i<featureValues.length; i++){
////				if( i != trainSet.getLabelIndex()){
////					featureValues[i] = (featureValues[i] - featureMean[i]) / featureVar[i];
////				}
////			}
////		}
////		
////	}
// 
  static void normalizeDataSandS(DataSet trainSet, DataSet testSet) throws Exception {

		int noOfFeatures = trainSet.getFeatureSize();
		double[] featureMin = new double[noOfFeatures];
		double[] featureMax = new double[noOfFeatures];
		
		for(int i = 0; i<noOfFeatures; i++){
			featureMin[i] = Double.POSITIVE_INFINITY;
			featureMax[i] = Double.NEGATIVE_INFINITY;
		}
		
		for(int i = 0; i< noOfFeatures; i++){
			
			for(Data d: trainSet.getData()){
				
				if(d.getFeatureValueAtIndex(i) > featureMax[i]){
					featureMax[i] = d.getFeatureValueAtIndex(i);
				}
				if(d.getFeatureValueAtIndex(i) < featureMin[i]){
					featureMin[i] = d.getFeatureValueAtIndex(i);
				}
				
			}
			
			for(Data d: testSet.getData()){
				
				if(d.getFeatureValueAtIndex(i) > featureMax[i]){
					featureMax[i] = d.getFeatureValueAtIndex(i);
				}
				if(d.getFeatureValueAtIndex(i) < featureMin[i]){
					featureMin[i] = d.getFeatureValueAtIndex(i);
				}
				
			}
			
		}
		
		for(Data d : trainSet.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				featureValues[i] = (featureValues[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
			}
		}
		
		for(Data d : testSet.getData()){
			double[] featureValues = d.getFeatureValues();
			
			for(int i = 0; i<featureValues.length; i++){
				featureValues[i] = (featureValues[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
			}
		}
		
		
		
	}
//
// static void normalizeData(DataSet trainSet, DataSet testSet) throws Exception {
//		
//		int noOfFeatures = trainSet.getFeatureSize();
//		double[] featureMean = new double[noOfFeatures];
//		
//		int trainDataSize = trainSet.getData().size();
//		int testDataSize = testSet.getData().size();
//		
//		for(int i = 0; i< noOfFeatures; i++){
//			
//			for(Data d: trainSet.getData()){
//				featureMean[i] += d.getFeatureValueAtIndex(i); 
//			}
//			
//			for(Data d: testSet.getData()){
//				featureMean[i] += d.getFeatureValueAtIndex(i);
//			}
//			
//		}
//		
//		for(int i = 0; i< noOfFeatures; i++){
//			featureMean[i] /= (trainDataSize + testDataSize);
//		}
//		
//		double[] featureVar = new double[noOfFeatures];
//		
//		
//		for(int i = 0; i< noOfFeatures; i++){
//			
//			for(Data d: trainSet.getData()){
////				featureVar[i] += d.getFeatureValue(i); 
//				featureVar[i] += Math.pow(d.getFeatureValueAtIndex(i) - featureMean[i],2);
//			}
//			
//			for(Data d: testSet.getData()){
////				featureVar[i] += d.getFeatureValue(i);
//				featureVar[i] += Math.pow(d.getFeatureValueAtIndex(i) - featureMean[i],2);
//			}
//			
//		}
//		
//		for(int i = 0; i< featureVar.length; i++){
//			featureVar[i] = Math.sqrt(featureVar[i]);
//		}
//		
//		for(Data d : trainSet.getData()){
//			double[] featureValues = d.getFeatureValues();
//			
//			for(int i = 0; i<featureValues.length; i++){
//				featureValues[i] = (featureValues[i] - featureMean[i]) / featureVar[i];
//			}
//		}
//		
//		for(Data d : testSet.getData()){
//			double[] featureValues = d.getFeatureValues();
//			
//			for(int i = 0; i<featureValues.length; i++){
//				featureValues[i] = (featureValues[i] - featureMean[i]) / featureVar[i];
//			}
//		}
//		
//	}
  
  public static void normalizeData(DataSet trainingData, DataSet testData) throws Exception {
		double[] featureValueMax = null;
		double[] featureValueMin = null;
		if (trainingData != null && trainingData.getDataSize() > 0) {
			int featureSize = trainingData.getFeatureSize();
			featureValueMax = new double[featureSize];
			featureValueMin = new double[featureSize];
			for (int i = 0; i < featureSize; i++) {
				featureValueMax[i] = Double.NEGATIVE_INFINITY;
				featureValueMin[i] = Double.POSITIVE_INFINITY;
			}

		}
		updateFeatureMinMaxValues(featureValueMax, featureValueMin, trainingData);
		updateFeatureMinMaxValues(featureValueMax, featureValueMin, testData);
		shiftAndScaleNormalize(trainingData, featureValueMin, featureValueMax);
		shiftAndScaleNormalize(testData, featureValueMin, featureValueMax);
	}

	private static void updateFeatureMinMaxValues(double[] featureValueMax, double[] featureValueMin, DataSet dataSet)
			throws Exception {
		if (dataSet != null && dataSet.getDataSize() > 0) {
			for (Data data : dataSet.getData()) {
				for (int featureIndex = 0; featureIndex < dataSet.getFeatureSize(); featureIndex++) {
					double value = data.getFeatureValueAtIndex(featureIndex);
					if (value > featureValueMax[featureIndex]) {
						featureValueMax[featureIndex] = value;
					}
					if (value < featureValueMin[featureIndex]) {
						featureValueMin[featureIndex] = value;
					}
				}
			}
		}
	}

	private static DataSet shiftAndScaleNormalize(DataSet dataSet, double[] featureValueMin, double[] featureValueMax)
			throws Exception {
		if (dataSet != null && dataSet.getDataSize() > 0) {
			for (Data data : dataSet.getData()) {
				double[] featureValues = data.getFeatureValues();
				for (int feature = 0; feature < featureValues.length; feature++) {
					featureValues[feature] = (featureValues[feature] - featureValueMin[feature])
							/ (featureValueMax[feature] - featureValueMin[feature]);
				}
			}
		}
		return dataSet;
	}
}

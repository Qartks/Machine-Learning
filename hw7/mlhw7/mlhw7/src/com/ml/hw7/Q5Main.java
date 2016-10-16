package com.ml.hw7;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Q5Main {

	public static void main(String[] args) throws Exception{
		int f = 5;
		makeNewDataSet(f);
		performKNNWithImportantFeatures(f);
	}

	private static void performKNNWithImportantFeatures(int f) throws Exception {
		DataInputer.noOFFeatures = 5;
		DataSet allData = DataInputer.getData("C:/Users/eesha_000/Downloads/ML/filteredData.txt");
		errorWithKFold(10, allData);
		
	}

	private static void makeNewDataSet(int f) throws Exception {
		DataInputer.noOFFeatures = 57;
		DataSet allData = DataInputer.getData("C:/Users/eesha_000/Downloads/spambase.data");
		Kernel kernel = new EuclidDistance(0);
		
		Relief r = new Relief((int)allData.getFeatureSize(), allData, kernel);
		double[] weights = r.getImpFeatures();
		
		DataSet newDataSet = filterDataSetWithImportFeatures(allData,weights, f);
		
		DataInputer.writeDataToFile("C:/Users/eesha_000/Downloads/ML/filteredData.txt", newDataSet);
	}

	private static DataSet filterDataSetWithImportFeatures(DataSet allData, double[] weights, int k) {
		DataSet newDataSet = new DataSet(k);
		List<FeatureValue> listOfFeats = getListOfFeatures(weights, k);
		for(Data d : allData.getData()){
			Data dNew = new Data(k);
			int i = 0;
			for(FeatureValue j : listOfFeats){
				dNew.setFeatureValueAtIndex(i, d.getFeatureValueAtIndex(j.featureId));
				i++;
			}
			dNew.setLabelValue(d.getLabelValue());
			newDataSet.addData(dNew);
		}
		return newDataSet;
	}

	private static List<FeatureValue> getListOfFeatures(double[] weights, int k) {
		List<FeatureValue> list = new ArrayList<FeatureValue>();
		List<FeatureValue> listK = new ArrayList<FeatureValue>();
		
		for(int i = 0; i < weights.length; i++){
			FeatureValue f = new FeatureValue(i, weights[i]);
			list.add(f);
		}
		Collections.sort(list, new Comparator<FeatureValue>() {

			@Override
			public int compare(FeatureValue o1, FeatureValue o2) {
				return Double.compare(o1.distance, o2.distance);
			}
		});
		
		for(int i = 0; i < k ; i++){
			listK.add(list.get(i));
		}
		
		System.out.println(listK.toString());
		return listK;
	}

	private static void errorWithKFold(int k, DataSet dataSet) throws Exception {
		int dataPerFold = dataSet.getDataSize()/ k;
		Collections.shuffle(dataSet.getData());
		double avgTestAcc = 0;
		
		for(int fold=0; fold< k ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
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
			System.out.println("Fold " + (fold + 1) + ":");
//			Kernel kernel = new GaussianKernel(10, 1.75d);
			Kernel kernel = new EuclidDistance(0);
			KNNImplementation knn = new KNNImplementation(1, 0, false, trainingData, kernel);
			avgTestAcc += knn.execute(testData);
		}
		System.out.println("Average Testing Accuracy: " + (1- avgTestAcc/k) * 100);
	}

}

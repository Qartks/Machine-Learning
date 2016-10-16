package com.ml.hw4;

import java.util.ArrayList;
import java.util.List;


public class BaggingImplementation {

	private DataSet2 allDataSet;
	private List<TreeNode> decisionTreeRoots =  new ArrayList<TreeNode>();
	
	public BaggingImplementation(DataSet2 dataSet) {
		this.allDataSet = dataSet;
	}
	
	public void train(int baggingNum) throws Exception {
		for(int i=0; i<baggingNum; i++) {
			DataSet2 trainingDataSet = getTrainingDataSetForBagging();
			TreeImplementation dT = new TreeImplementation(7,0.03,39);
			decisionTreeRoots.add(dT.execute(trainingDataSet));
		}
	}
	
	public double test(DataSet2 testDataSet) throws Exception {
		TreeImplementation dT = new TreeImplementation(7,0.03,39);
		double error = 0;
		for(Data2 data : testDataSet.getData()) {
			double predictedLabel = 0;
			double actualLabel = data.labelValue() == 0 ? -1 : 1;
			
			for(TreeNode tree : decisionTreeRoots) {
				double predictionByTree = dT.predictedValue(tree, data);
				predictedLabel+= predictionByTree == 0 ? -1 : 1;
			}
			
			predictedLabel = Math.signum(predictedLabel);
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		return error/testDataSet.dataSize();
	}

	private DataSet2 getTrainingDataSetForBagging() throws Exception {
		DataSet2 trainingDataSet = new DataSet2(allDataSet.getLabelIndex(), allDataSet.getFeatures());
		List<Data2> trainingData = randomSampleWithReplacement(allDataSet.getData(), 100);
		addDataToDataSet(trainingDataSet, trainingData);
		return trainingDataSet;
	}
	
	private List<Data2> randomSampleWithReplacement(List<Data2> data, int n) {
		List<Data2> list = new ArrayList<Data2>();
		for(int i = 0; i < n; i++){
			int ind = (int) (Math.random() * data.size());
			list.add(data.get(ind));
		}
		return list;
	}

	private void addDataToDataSet(DataSet2 dataset, List<Data2> datas) throws Exception {
		for(Data2 data : datas) {
			dataset.addData(data);
		}
	}
}

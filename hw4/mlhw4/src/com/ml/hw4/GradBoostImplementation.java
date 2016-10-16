package com.ml.hw4;

import java.util.ArrayList;
import java.util.List;


class GradBoostImplementation {
	
	public GradBoostImplementation() {
		
	}

	public void boooostAndTrain(int n, DataSet2 trainingData, DataSet2 testData) throws Exception {
		
		TreeImplementation regressionTree = new TreeImplementation(2, .15, 40);
		List<TreeNode> list = new ArrayList<TreeNode>();
		
		for(Data2 d : trainingData.getData()){
			d.setOriginalLabel(d.labelValue());
		}
		
		for(Data2 d : testData.getData()){
			d.setOriginalLabel(d.labelValue());
		}
		
		for(int i = 0; i < n; i++){
			TreeNode root = regressionTree.execute(trainingData);
			updateLabels(trainingData, root, regressionTree);
			
			list.add(root);
			errorResults(trainingData, regressionTree, list);
		}

		System.out.print("Train -> ");
		errorResults(trainingData, regressionTree, list);
		System.out.print("Test -> ");
		errorResults(testData, regressionTree, list);
		
	}

	private void updateLabels(DataSet2 trainingData, TreeNode root, TreeImplementation rT) throws Exception {
		for(Data2 d : trainingData.getData()){
			double predictVal = rT.predictedValue(root, d);
			double lErr = d.labelValue() - predictVal;
			d.setLabelValue(lErr);
		}
	}
	
	public static void errorResults(DataSet2 testData, TreeImplementation regressionTree, List<TreeNode> list) throws Exception {
		double diffSum = 0;
		
		for(Data2 data : testData.getData()){
			double perdictedValue = 0;
			for(TreeNode tn : list) {
				perdictedValue += regressionTree.predictedValue(tn,data);
			}
			diffSum += Math.pow((data.getOriginalLabel() - perdictedValue), 2);
		}
		
		double mse = diffSum/testData.getData().size();
		
		System.out.println("Mean Square Error:" + mse);
	}

}

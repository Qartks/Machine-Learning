package com.ml.hw2;

import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;


public class TreeImplementation {
	
	
	public TreeImplementation(int maxDepth, double minGain, int minDataPerNode) {
		TreeConstants.MAX_DEPTH = maxDepth;
		TreeConstants.MIN_GAIN= minGain;
		TreeConstants.MIN_DATA= minDataPerNode;
	}
	
	public TreeNode execute(DataSet dataset) throws Exception {
		TreeNode root = new TreeNode(dataset, 0);
		
		buildTree(root);
		
		return root;
	}

	public TreeNode buildTree(TreeNode root) throws Exception {
		
		Queue<TreeNode> nodesQueue = new ArrayDeque<TreeNode>();
		Map<Integer, Boolean> featureHistory = new HashMap<Integer, Boolean>();
		
		nodesQueue.add(root);
		
		while(!nodesQueue.isEmpty()) {
			splitNode(nodesQueue.remove(), nodesQueue, featureHistory);
		}
		
		return root;
	}
	
	private void splitNode(TreeNode node, Queue<TreeNode> nodesQueue, Map<Integer, Boolean> featureHistory) throws Exception {
		NodeStats stats = node.getStats();
		DataSet dataSet = node.getDataSet();
		if(node.isLeaf() || node.depth() == TreeConstants.MAX_DEPTH || (dataSet.isClassificationTask() && hasOneOfaKind(stats)) || 
				stats.getTotalData() <= TreeConstants.MIN_DATA) {
			node.setLeaf();
			return;
		}
		double maxInfoGain = Double.NEGATIVE_INFINITY;
		SplitNode splitResult = null;
		SplitNode bestResult = null;
		for(int i=0; i< dataSet.getFeatures().size(); i++) {
			if(i != dataSet.getLabelIndex()) {
				if(dataSet.getFeatures().get(i).isNominal() && !featureHistory.containsKey(i)) {
					splitResult = splitCategoricalFeature(i, node);
				} else {
					splitResult = splitNumericalFeature(i, node);
				}
				if(splitResult == null) {
					continue;
				}
				if(splitResult.getInfoGain() > maxInfoGain) {
					maxInfoGain = splitResult.getInfoGain();
					bestResult = splitResult;
				}
			}
		}
		
		if(bestResult == null || (bestResult != null && bestResult.getInfoGain() < TreeConstants.MIN_GAIN)) {
			node.setLeaf();
			return;
		}
		
//		System.out.println( "Best Feature Index" +  bestResult.getSplitFeatureIndex() + " Best FeatureValue: " + bestResult.getSplitFeatureValue() +" Info Gain: " + bestResult.getInfoGain());
		featureHistory.put(bestResult.getSplitFeatureIndex(), true);
		
		node.setNoBranch(bestResult.getNoBranch());
		node.setYesBranch(bestResult.getYesBranch());
		node.setSplitFeatureIndex(bestResult.getSplitFeatureIndex());
		if(node.getDataSet().getFeature(bestResult.getSplitFeatureIndex()).isNumerical()) {
			node.setSplitFeatureValue(bestResult.getSplitFeatureValue());
		}
		
		
		nodesQueue.add(bestResult.getNoBranch());
		nodesQueue.add(bestResult.getYesBranch());
	}

	private SplitNode splitCategoricalFeature(int featureIndex, TreeNode node) throws Exception {
		
		SplitNode result = new SplitNode();
		DataSet dataSet = node.getDataSet();
		double oldEntropy = Entropy.calculate(node);
			
		TreeNode noBranch = new TreeNode(dataSet.getFeatures(), node.depth()+1, node.getDataSet().getLabelIndex(), dataSet.classNum());
		TreeNode yesBranch = new TreeNode(dataSet.getFeatures(), node.depth()+1, node.getDataSet().getLabelIndex(), dataSet.classNum());
		
		for(Data data : dataSet.getData()) {
			if(data.getFeatureValue(featureIndex) == 0d) {
				noBranch.addData(data);
			} else {
				yesBranch.addData(data);
			}
		}
		
		if(noBranch.getStats().getTotalData() == 0 || yesBranch.getStats().getTotalData() == 0) {
			node.setLeaf();
			node.setSplitFeatureIndex(featureIndex);
			return null;
		}
		
		double entropyNoBranch = hasOneOfaKind(noBranch.getStats()) ? 0 : Entropy.calculate(noBranch);
		double entropyYesBranch = hasOneOfaKind(yesBranch.getStats()) ? 0 : Entropy.calculate(yesBranch);
		double combinedEntrypy = (entropyNoBranch * ((double)noBranch.getStats().getTotalData()/node.getStats().getTotalData()))+ 
				(entropyYesBranch * ((double)yesBranch.getStats().getTotalData()/node.getStats().getTotalData()));
		double infoGain = oldEntropy - combinedEntrypy;
		
		result.setInfoGain(infoGain);
		result.setYesBranch(yesBranch);
		result.setNoBranch(noBranch);
		result.setSplitFeatureIndex(featureIndex);
		
		return result;
	}

	
	private SplitNode splitNumericalFeature(int featureIndex, TreeNode node) throws Exception {
		
		sortFeatures(node.getDataSet(), featureIndex);
		
		List<Data> dataset = node.getDataSet().getData();
		double initEntropy = Entropy.calculate(node);
		FeatureThreshold bestThreshold = optimalThreshold(node, initEntropy, featureIndex);
		
		if(!bestThreshold.isThresholdExist()) {
			return null;
		}
		
		
		
		TreeNode noBranch = new  TreeNode(node.getDataSet().getFeatures(), node.depth()+1, node.getDataSet().getLabelIndex(), node.getDataSet().classNum());
		TreeNode yesBranch = new  TreeNode(node.getDataSet().getFeatures(), node.depth()+1, node.getDataSet().getLabelIndex(), node.getDataSet().classNum());
		
		for(Data data : dataset) {
			if(data.getFeatureValue(featureIndex) <= bestThreshold.getThreshold()) {
				noBranch.addData(data);
			} else {
				yesBranch.addData(data);
			}
		}
		
		if(noBranch.getStats().getTotalData() == 0 || yesBranch.getStats().getTotalData() == 0) {
			node.setLeaf();
			node.setSplitFeatureIndex(featureIndex);
			return null;
		}
		
		SplitNode result = new SplitNode();
		result.setInfoGain(bestThreshold.getInfoGain());
		result.setNoBranch(noBranch);
		result.setYesBranch(yesBranch);
		result.setSplitFeatureIndex(featureIndex);
		result.setSplitFeatureValue(bestThreshold.getThreshold());
		
		return result;
		
	}

	private FeatureThreshold optimalThreshold(TreeNode node, double initEntropy, int featureIndex) throws Exception {

		FeatureThreshold bestThreshold = null;
		NodeStats stats = node.getStats();
		
		double maxInfoGain = Double.NEGATIVE_INFINITY;
		double leftSum = 0;
		double threshold = 0;
		boolean thresholdExist = false;
		
		List<Data> dataset = node.getDataSet().getData();
		
		for(int i=0; i< dataset.size()-1; i++) {
			
			Data data1 = dataset.get(i);
			Data data2 = dataset.get(i+1);
			leftSum+= data1.labelValue();
			
			if(data1.getFeatureValue(featureIndex) != data2.getFeatureValue(featureIndex)) {
				
				threshold = ((double) data1.getFeatureValue(featureIndex) + data2.getFeatureValue(featureIndex))/2;
				
				double[] total = node.getDataSet().getLeftData(node.getDataSet().dataSize()-1, featureIndex);
				double[] left = node.getDataSet().getLeftData(i, featureIndex);
				double[] right = node.getDataSet().getRightData(i, featureIndex);
				
//				double origMSE = Entropy.calculateMSE(total, node.getStats().getTotalValue());
				FeatureThreshold currentThreshold = new FeatureThreshold(stats, i, leftSum, threshold,
						initEntropy, node.getDataSet().isClassificationTask(), left, right, node);
				
				double infoGain = currentThreshold.getInfoGain();
				
				if(infoGain > maxInfoGain) {
					maxInfoGain = infoGain;
					bestThreshold = currentThreshold;
					thresholdExist = true;
				}
			}
		}
		
		if(!thresholdExist) {
			return new  FeatureThreshold();
		}
		
		return bestThreshold;
	}

	private void sortFeatures(DataSet dataSet,int featureIndex) {
		if(dataSet != null) {
			Collections.sort(dataSet.getData(), new Comparator<Data>() {

				@Override
				public int compare(Data arg0, Data arg1) {
					Double value1 = new Double(arg0.getFeatureValue(featureIndex));
					Double value2 = new Double(arg1.getFeatureValue(featureIndex));
					return value1.compareTo(value2);
				}
			});
		}
	}
	

	private boolean hasOneOfaKind(NodeStats stats) {
		int[] labelsPerClass = stats.getLabelPerClass();
		for(int labelCount : labelsPerClass) {
			if(labelCount == stats.getTotalData()) {
				return true;
			}
		}
		return false;
	}
	
	
	public double predictedValue(TreeNode node, Data data) throws Exception {
		if(node.isLeaf()) {
			return node.getLeafValue();
		}
		if(data.getFeature(node.getSplitFeatureIndex()).isNominal()) {
			if(data.getFeatureValue(node.getSplitFeatureIndex())==0) {
				return predictedValue(node.getNoBranch(), data);
			} else {
				return predictedValue(node.getYesBranch(), data);
			}
		} else {
			if(data.getFeatureValue(node.getSplitFeatureIndex()) < node.getSplitFeatureValue()) {
				return predictedValue(node.getNoBranch(), data);
			} else {
				return predictedValue(node.getYesBranch(), data);
			}
		}
	}
	

}

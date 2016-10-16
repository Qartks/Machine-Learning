package ml.hw1;

import java.util.List;

public class FeatureThreshold {
	
	private double threshold;
	private double labelSumOnLeft;
	private int noOfLabelsOnLeft;
	private NodeStats stats;
	private double infoGain;
	private boolean thresholdExist;
	
	
	public FeatureThreshold(NodeStats stats, int rowIndex, double leftSum, double threshold,
			double oldEntropy, boolean isClassificationTask, double[] left, double[] right, TreeNode node) throws Exception {
		this.threshold = threshold;
		this.labelSumOnLeft = leftSum;
		this.noOfLabelsOnLeft = rowIndex+1;
		this.stats = stats;
		this.infoGain = getInfoGain(oldEntropy, isClassificationTask, left, right, node);
		thresholdExist = true;
	}

	public FeatureThreshold() {
		thresholdExist = false;
	}

	private double getInfoGain(double oldEntropy, boolean isClassificationTask
			, double[] left, double[] right , TreeNode node) throws Exception {
		if(isClassificationTask) {
			return getInfoGainForClassification(oldEntropy);
		}
		return getInfoGainForRegression(oldEntropy, left, right, node);
	}
	
	private double getInfoGainForRegression(double oldEntropy, double[] left, double[] right, TreeNode node) throws Exception {
//		double totalValueOfNode = stats.getTotalValue();
//		double totalValueOnRight = totalValueOfNode - noOfOnesOnLeft;
//		int noOfDataOnRight = this.stats.getTotalData() - noOfDataOnLeft;
//		double temp = Math.pow(noOfOnesOnLeft, 2)/noOfDataOnLeft + Math.pow(totalValueOnRight, 2)/noOfDataOnRight -
//				Math.pow(totalValueOfNode, 2)/this.stats.getTotalData();
//		

//		double leftMSE = Entropy.calculateMSE(left, noOfOnesOnLeft);
//		double rightMSE = Entropy.calculateMSE(right, stats.getTotalValue() - noOfOnesOnLeft);
		
		
//		return (stats.getTotalValue()/stats.getTotalData()) - (left.length * leftMSE + right.length * rightMSE);
//		return (stats.getTotalValue()/stats.getTotalData()) 
//				- (noOfOnesOnLeft/left.length*leftMSE + (stats.getTotalValue() - noOfOnesOnLeft)*rightMSE/right.length);
		
//		return (leftMSE*left.length/stats.getTotalData() + rightMSE*right.length/stats.getTotalData());
		
		double totalValueOfNode = stats.getTotalValue();
		double totalValueOnRight = totalValueOfNode - labelSumOnLeft;
		int noOfDataOnRight = this.stats.getTotalData() - noOfLabelsOnLeft;
		double meanLeft = labelSumOnLeft/noOfLabelsOnLeft;
		double meanRight = totalValueOnRight/noOfDataOnRight;
		double leftMSE = 0;
		double rightMSE = 0;
		List<Data> datas = node.getDataSet().getData();
		int counter = 0;
		for(Data data : datas) {
			if(counter < noOfLabelsOnLeft) {
				leftMSE+= Math.pow(meanLeft - data.labelValue(), 2);
			} else {
				rightMSE+= Math.pow(meanRight - data.labelValue(), 2);
			}
			counter++;
		}
		leftMSE/= noOfLabelsOnLeft;
		rightMSE/= noOfDataOnRight;
		double combinedReduction = leftMSE * noOfLabelsOnLeft/stats.getTotalData() + rightMSE * noOfDataOnRight/stats.getTotalData();
		//return oldEntropy - (leftMSE+rightMSE);
		return oldEntropy - combinedReduction;
		
	}

	private double getInfoGainForClassification(double oldEntropy) {
		
		double totalNumberOfOnes = this.stats.getLabelPerClass()[1];
		double noOfOnesOnRight = totalNumberOfOnes - labelSumOnLeft;
		double noOfDataOnRight = this.stats.getTotalData() - noOfLabelsOnLeft;
		double noOfZeroOnLeft = noOfLabelsOnLeft - labelSumOnLeft;
		double noOfZeroOnRight = noOfDataOnRight - noOfOnesOnRight;
		
		double leftEntropy = getEntropy(noOfZeroOnLeft, labelSumOnLeft, noOfLabelsOnLeft);
		double rightEntropy = getEntropy(noOfZeroOnRight, noOfOnesOnRight, noOfDataOnRight);
		double combinedEntropy = ((double)noOfLabelsOnLeft)/stats.getTotalData()* leftEntropy +
				((double)noOfDataOnRight)/stats.getTotalData()* rightEntropy;
		
		return oldEntropy - combinedEntropy;
		
	}
	
	private double getEntropy(double noOfZero, double noOfOne, double totalData) {
		
		if(noOfZero == 0 || noOfOne == 0) {
			return 0;
		}
		
		double probability1 = noOfZero/totalData; 
		double probability2 = noOfOne/totalData;
		double entropy = probability1*Math.log(probability1)/Math.log(2) + probability2*Math.log(probability2)/Math.log(2);
		
		return -entropy;
	}
	
	
	public double getThreshold() {
		return threshold;
	}


	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}


	public double getInfoGain() {
		return infoGain;
	}


	public void setInfoGain(double infoGain) {
		this.infoGain = infoGain;
	}


	public boolean isThresholdExist() {
		return thresholdExist;
	}


	public void setThresholdExist(boolean thresholdExist) {
		this.thresholdExist = thresholdExist;
	}

}

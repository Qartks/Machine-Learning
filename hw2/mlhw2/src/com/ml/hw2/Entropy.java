package com.ml.hw2;

public class Entropy {

	public static double calculate(TreeNode node) throws Exception {
		
		if(!node.getDataSet().isClassificationTask()) {
			
			double mean = node.getStats().getTotalValue()/
					node.getStats().getTotalData();
			
			double err = 0;
			
			for(Data data : node.getDataSet().getData()) {
				err+= Math.pow(data.labelValue() 
						- mean, 2);
			}
			return err/node.getDataSet().dataSize();
		}
		
		NodeStats stats = node.getStats();
		int totalData = stats.getTotalData();
		int [] labelCountPerClass = stats.getLabelPerClass();
		double entropy = 0;
		for(int classs=0; classs < labelCountPerClass.length; classs++) {
			int labelCount = labelCountPerClass[classs];
			double probability = ((double)labelCount/totalData);
			double temp = probability * Math.log(probability) / Math.log(2);
			entropy+= temp;
		}
		
		return -entropy;
	}
	
	public static double calculateMSE(double[] data, double sum){
		
		double average = sum/data.length;
		double squareErrSum = 0;
		for(int i = 0; i<data.length; i++){
			squareErrSum += Math.pow((average - data[i]), 2);
		}
		return squareErrSum/data.length;
	}
}

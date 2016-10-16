package com.ml.hw7;

public class Q3MainB {

	public static void main(String[] args) throws Exception {
		
		DataInputer.noOFFeatures = 4;
		DataSet trainingData = DataInputer.getData("C:/Users/eesha_000/Downloads/twoSpirals.txt");
		
//		Kernel kernel = new GaussianKernel(trainingData.getDataSize(), 1.75d);
		Kernel kernel = new DotProduct(trainingData.getDataSize());
		DualPerceptron dp = new DualPerceptron(kernel,trainingData.getDataSize());
		dp.train(trainingData);
	}

}

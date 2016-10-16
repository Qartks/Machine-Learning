package com.ml.hw7;

public class Q3MainA {

	public static void main(String[] args) throws Exception {
		
		DataInputer.noOFFeatures = 4;
		DataSet trainingData = DataInputer.getData("C:/Users/eesha_000/Downloads/perceptronData.txt");
		
		Kernel kernel = new GaussianKernel(trainingData.getDataSize(), 1.75d);
		DualPerceptron dp = new DualPerceptron(kernel,trainingData.getDataSize());
		dp.train(trainingData);
	}

}

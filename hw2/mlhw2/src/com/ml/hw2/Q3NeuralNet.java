package com.ml.hw2;

import java.util.ArrayList;
import java.util.List;

public class Q3NeuralNet {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		AutoEncoder.initializeInputData();
		AutoEncoder.learn();
		
		NNAutoencoder encoder = new NNAutoencoder(9, 4, 8);
		List<double[]> trainingData = new ArrayList<double[]>();;
		encoder.initialize(trainingData);
		encoder.train(trainingData, 3);
		encoder.test(trainingData);

	}

}

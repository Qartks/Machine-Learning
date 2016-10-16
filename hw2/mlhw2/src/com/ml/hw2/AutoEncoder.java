package com.ml.hw2;

import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class AutoEncoder {
	
	private static int noOfInput = 8;
	private static int noOfOutput = 8;
	private static int noOfHidden = 3;
	
	private static double[][] trainingData = new double[noOfInput][noOfInput];
	private static double[][] outputData = new double[noOfOutput][noOfOutput];
	
	private static double[][] inputToHiddenWeights = new double[8][3];
	private static double[][] hiddenToOutputWeights = new double[3][8];
	
	private static double[] inputToHiddenBias = new double[3];
	private static double[] hiddenToOutputBias = new double[8];
	
	private static double[] hiddenErrors = new double[3];
	private static double[] outputErrors = new double[8];;
	
	private static double[] hiddenNetInputs = new double[3];
	private static double[] outputNetInputs = new double[8];
	
	private static double[] outputValues = new double[8];
	private static double[] hiddenValues = new double[3];
	
	
	private static double LAMBDA = 0.002;
	
	private static int index = 0;
	
	public static void initializeData(){
		
		
		for(int i = 0; i< 8; i++){
			for(int j = 0; j< 3; j++){
				inputToHiddenWeights[i][j] = Math.random();
				hiddenToOutputWeights[j][i] = Math.random();
			}
		}
		
//		outputValues = new double[8];
//		hiddenValues = new double[3];
		
	}

	
	public static void learn(){
		
		// Until Some Condition
		//	For each Data tuple
		//
		//
		// 		Forward Feed
		//			Propagate to Hidden
		// 			Propagate to Output
		// 		Back Propagation
		//			Error in Output layer
		//			Error in Hidden Layer
		//
		//
		int iter = 1;
		initializeData();
		SimpleMatrix w1 = new SimpleMatrix(inputToHiddenWeights);
		System.out.println(w1.toString());
//		while(iter < 500000){
			for(int i = 0; i < noOfInput; i++){
				
				iter = 1;
				while(iter < 500000){
				double[] trainingTuple = trainingData[i];
				double[] outputTuple = outputData[i];
		
				index = i;
	
				propagateInputForward(trainingTuple);
				backwardPropagate();
				iter++;
				}
//				iter++;
				System.out.println(Arrays.toString(hiddenValues) + " -> " + Arrays.toString(outputValues));
//			}
		}
		
		SimpleMatrix w = new SimpleMatrix(inputToHiddenWeights);
		System.out.println(w.toString());
		
//		testResult();
		
		
	}

	private static void testResult() {
		for(int i = 0; i< noOfInput; i++){

			double[] trainingTuple = trainingData[i];
			double[] outputTuple = outputData[i];
			
			index = i;
			
			propagateInputForward(trainingTuple);
//			System.out.println("Hi" + i);
//			backwardPropagate();
			
			System.out.println(Arrays.toString(trainingTuple) + " -> " + Arrays.toString(hiddenValues) + " -> " + Arrays.toString(outputValues));
		}
	}


	private static boolean terminatingCondition() {
		
		return false;
	}


	private static void propagateInputForward(double[] trainingTuple) {
		
		propagateToHiddenLayer(trainingTuple);
		propagateToOutputLayer();
		
	}
	

	private static void propagateToHiddenLayer(double[] trainingTuple) {
		
//		System.out.println(" -> " + Arrays.toString(trainingTuple) + " -> ");
		
		for(int i = 0; i<hiddenValues.length; i++){
			double netInput = 0;
//			double[] weights = inputToHiddenWeights[i];
//			System.out.println(Arrays.toString(weights));
			for(int j = 0; j < trainingTuple.length; j++){
				netInput += (inputToHiddenWeights[j][i]*trainingTuple[j]);
			}
			
			netInput += inputToHiddenBias[i];
			double result = 1 / (1 + Math.exp(-netInput));
			
			hiddenValues[i] = result;
			hiddenNetInputs[i] = result;
		}
		
	}


	private static void propagateToOutputLayer() {
		
		for(int i = 0; i< outputValues.length; i++){
			double netInput = 0;
			
//			double[] weights = hiddenToOutputWeights[i];
//			System.out.println(Arrays.toString(weights));
			for(int j = 0; j < hiddenValues.length; j++){
				netInput += (hiddenToOutputWeights[j][i] * hiddenValues[j]);
			}
			netInput += hiddenToOutputBias[i];
			double result = 1 / (1 + Math.exp(-netInput));
			
			outputValues[i] = result;
			outputNetInputs[i] = result;
			
		}
	}


	private static void backwardPropagate() {
		
		calculateError();
		updateWeights();
		updateBais();
		
	}

	private static void calculateError() {
		
		for(int i = 0; i < outputErrors.length; i++){
			
			double targetOutput = outputData[index][i];
			double predictedValue = outputValues[i];
			
			double error = predictedValue * (1 - predictedValue) * (targetOutput - predictedValue);
			
			outputErrors[i] = error;
		}
		
		for (int i = 0; i < hiddenErrors.length; i++) {
			
			double predictedValue = hiddenValues[i];
			double sumError = 0;
			
			for(int j = 0; j< outputErrors.length; j++){
				sumError += outputErrors[j] * hiddenToOutputWeights[i][j];
			}
			
			double error = predictedValue * (1 - predictedValue) * sumError;
			
			hiddenErrors[i] = error;
		}

		
	}


	private static void updateWeights() {
		
			for(int i = 0; i< inputToHiddenWeights.length; i++){
				for(int j = 0; j< inputToHiddenWeights[0].length; j++){
					double updatedWeight = inputToHiddenWeights[i][j] + LAMBDA * hiddenErrors[j] * hiddenValues[j];
					inputToHiddenWeights[i][j] = updatedWeight;
				}
			}
			
			for(int i = 0; i< hiddenToOutputWeights.length; i++){
				for(int j = 0; j< hiddenToOutputWeights[0].length; j++){
					double updatedWeight = hiddenToOutputWeights[i][j] + LAMBDA * outputErrors[i] * outputValues[i];
					hiddenToOutputWeights[i][j] = updatedWeight;
				}
			}
		
	}
	

	private static void updateBais() {
		
		for(int i = 0; i< inputToHiddenBias.length; i++){
			double newBias = inputToHiddenBias[i] + LAMBDA * hiddenErrors[i];
			inputToHiddenBias[i] = newBias;
		}
		
		for(int i = 0; i< hiddenToOutputBias.length; i++){
			double newBias = hiddenToOutputBias[i] + LAMBDA * outputErrors[i];
			hiddenToOutputBias[i] = newBias;
		}
		
	}


	public static void initializeInputData() {
		// TODO Auto-generated method stub
		
		for(int i = 0; i< noOfInput; i++){
			for(int j = 0; j< noOfInput; j++){
				trainingData[i][j] = ((i == j) ? 1 : 0);
				outputData[i][j] = ((i == j) ? 1 : 0);
			}
		}
	}
	
}

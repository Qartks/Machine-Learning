package com.ml.hw2;

import java.util.Arrays;
import java.util.List;

public class NNAutoencoder {
	
	private NNNode[] inputLayer;
	private NNNode[] hiddenLayer;
	private NNNode[] outputLayer;
	private double[][] inputToHiddenWeights;
	private double[][] hiddenToOutputWeights;
	private int noOfInputNode;
	private int noOfHiddenNode;
	private int noOfOutputNode;
	
	public NNAutoencoder(int inputNodeNum, int hiddenNodeNum, int outputNodeNum) {
		this.noOfInputNode = inputNodeNum;
		this.noOfHiddenNode = hiddenNodeNum;
		this.noOfOutputNode = outputNodeNum;
		inputLayer = new NNNode[inputNodeNum];
		hiddenLayer = new NNNode[hiddenNodeNum];
		outputLayer = new NNNode[outputNodeNum];
		inputToHiddenWeights = new double[inputNodeNum][hiddenNodeNum];
		hiddenToOutputWeights = new double[hiddenNodeNum][outputNodeNum];

	}
	
	public void initialize(List<double[]> trainingData){
		
		trainingData.add(new double[]{1,0,0,0,0,0,0,0});
		trainingData.add(new double[]{0,1,0,0,0,0,0,0});
		trainingData.add(new double[]{0,0,1,0,0,0,0,0});
		trainingData.add(new double[]{0,0,0,1,0,0,0,0});
		trainingData.add(new double[]{0,0,0,0,1,0,0,0});
		trainingData.add(new double[]{0,0,0,0,0,1,0,0});
		trainingData.add(new double[]{0,0,0,0,0,0,1,0});
		trainingData.add(new double[]{0,0,0,0,0,0,0,1});
		
		initializeNetworkWeights();
		initializeHiddenAndOutputLayerNodes();
	}
	
	private void initializeNetworkWeights() {
		for(int row=0; row < noOfInputNode; row++) {
			for(int col=0; col < noOfHiddenNode; col++) {
				if(col == 0) {
					inputToHiddenWeights[row][col] = 0;
				} else {
					inputToHiddenWeights[row][col] = Math.random();
				}
			}
		}
		for(int row=0; row < noOfHiddenNode; row++) {
			for(int col=0; col < noOfOutputNode; col++) {
				hiddenToOutputWeights[row][col] = Math.random();
			}
		}
	}
	
	private void initializeHiddenAndOutputLayerNodes() {
		for(int count=0; count < noOfHiddenNode; count++) {
			if(count == 0) {
				hiddenLayer[count] = new NNNode(1,1,0);
			} else {
				hiddenLayer[count] = new NNNode();
			}
		}
		for(int count=0; count < noOfInputNode; count++) {
			if(count == 0) {
				inputLayer[count] = new NNNode(1, 1, 0);
			} else {
				inputLayer[count] = new NNNode();
			}
		}
		for(int count=0; count < noOfOutputNode; count++) {
			outputLayer[count] = new NNNode();
		}
	}
	
	public void train(List<double[]> trainingDataSet, double lambda) {
		if(trainingDataSet != null && trainingDataSet.size() > 0) {
			int iterations = 0;
			while(true) {
				iterations++;
				for(double[] trainingDataPoint : trainingDataSet) {
					learn(trainingDataPoint, lambda);
				}
				if(iterations > 1000) {
					break;
				}
			}
		}
	}

	private void learn(double[] trainingDataPoint , double lambda) {
		importTrainingDataIntoNetwork(trainingDataPoint);
		updateHiddenNetworkNetInputAndOutPut();
		updateOutputLayerNetInputAndOutput();
		backPropagateError(lambda);
		
	}
	
	private void importTrainingDataIntoNetwork(double[] trainingDataPoint) {
		for(int count=0; count < trainingDataPoint.length; count++) {
			inputLayer[count+1].setNetInput(trainingDataPoint[count]);
			inputLayer[count+1].setOutput(trainingDataPoint[count]);
			outputLayer[count].settargetOutput(trainingDataPoint[count]);
		}
	}
	
	private void updateHiddenNetworkNetInputAndOutPut() {
		for(int hiddenRow = 1; hiddenRow < noOfHiddenNode; hiddenRow++) {
			double netInput = 0;
			for(int inputRow = 0; inputRow < noOfInputNode; inputRow++) {
				netInput+= inputLayer[inputRow].getOutput() * inputToHiddenWeights[inputRow][hiddenRow];
			}
			hiddenLayer[hiddenRow].setNetInput(netInput);
		}
	}
	
	private void updateOutputLayerNetInputAndOutput() {
		for(int outputRow = 0; outputRow < noOfOutputNode; outputRow++) {
			double netInput = 0;
			for(int hiddenRow = 0; hiddenRow < noOfHiddenNode; hiddenRow++) {
				netInput+= hiddenLayer[hiddenRow].getOutput() * hiddenToOutputWeights[hiddenRow][outputRow];
			}
			outputLayer[outputRow].setNetInput(netInput);
		}
	}
	
	private void backPropagateError(double lambda) {
		
		double[] errorOutputLayer = calculateOutputLayerError();
		double[] errorHiddenLayer = calculateHiddenLayerError(errorOutputLayer);
		updateHiddenToOutputLayerWeights(errorOutputLayer, errorHiddenLayer, lambda);
		updateInputToHiddenLayerWeights(errorOutputLayer, errorHiddenLayer, lambda);
	}

	private double[] calculateOutputLayerError() {
		double[] errorOutputLayer = new double[noOfOutputNode];
		for(int outputRow=0; outputRow < noOfOutputNode; outputRow++) {
			NNNode node = outputLayer[outputRow];
			errorOutputLayer[outputRow] = node.getGradiant() * node.getError();
		}
		return errorOutputLayer;
	}
	
	private double[] calculateHiddenLayerError(double[] errorOutputLayer) {
		double[] errorHiddenLayer = new double[noOfHiddenNode];
		
		for(int hiddenRow = 0; hiddenRow < noOfHiddenNode; hiddenRow++) {
			double error = 0;
			NNNode node = hiddenLayer[hiddenRow];
			for(int outputRow = 0; outputRow < noOfOutputNode; outputRow++) {
				error+= errorOutputLayer[outputRow] * hiddenToOutputWeights[hiddenRow][outputRow];
			}
			error*= node.getGradiant();
			errorHiddenLayer[hiddenRow] = error;
		}
		return errorHiddenLayer;
	}
	
	private void updateHiddenToOutputLayerWeights(double[] errorOutputLayer, double[] errorHiddenLayer, double lambda) {
		for(int hiddenRow = 0; hiddenRow < noOfHiddenNode; hiddenRow++) {
			NNNode node = hiddenLayer[hiddenRow];
			for(int outputRow = 0; outputRow < noOfOutputNode; outputRow++) {
				hiddenToOutputWeights[hiddenRow][outputRow]+= errorOutputLayer[outputRow] * node.getOutput() * lambda;
			}
		}
	}

	private void updateInputToHiddenLayerWeights(double[] errorOutputLayer, double[] errorHiddenLayer, double lambda) {
		for(int inputRow=0; inputRow < noOfInputNode; inputRow++) {
			NNNode node = inputLayer[inputRow];
			for(int hiddenRow=0; hiddenRow < noOfHiddenNode; hiddenRow++) {
				if(hiddenRow == 0) {
					continue;
				}
				inputToHiddenWeights[inputRow][hiddenRow]+= errorHiddenLayer[hiddenRow] * node.getOutput() * lambda;
			}
		}
	}
	
	private String printLayerOutput(NNNode[] layer, boolean hiddenLayer, boolean output) {
		StringBuilder sb = new StringBuilder();
		for(int hiddenRow=0; hiddenRow < layer.length; hiddenRow++) {
			if(hiddenLayer && hiddenRow == 0) {
				continue;
			}
			if(output){
				sb.append((layer[hiddenRow].getOutput()) < 0.3 ? 0 : 1);
			} else {
				sb.append(layer[hiddenRow].getOutput());
			}
			if( hiddenRow != layer.length-1)
				sb.append(", ");	
		}
		return sb.toString();
	}

	public void test(List<double[]> testDataSet) {
		for(double[] testDataPoint : testDataSet) {
			importTrainingDataIntoNetwork(testDataPoint);
			updateHiddenNetworkNetInputAndOutPut();
			updateOutputLayerNetInputAndOutput();
			System.out.println(Arrays.toString(testDataPoint) + " -> "+ printLayerOutput(hiddenLayer, true, false) +"    -> \t"+ printLayerOutput(outputLayer, false, true));
		}
	}

}

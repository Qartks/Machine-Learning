package com.ml.hw7;

import java.util.Arrays;
import java.util.List;

public class DualPerceptron {

	Kernel kernel;
	double[] alphas;
	
	public DualPerceptron(Kernel kernel, int dataSize) {
		this.kernel = kernel;
		alphas = new double[dataSize];
	}
	
	public void train(DataSet trainingData) throws Exception {
		int iteration = 0;
		while (true) {
			double changed = runIteration(trainingData);
			System.out.println("Iteration "+ iteration+++"  alpha change -> "+changed);
			if(changed == 0) {
				break;
			}
		}
	}

	private double runIteration(DataSet trainingData) throws Exception {
		double changeCount = 0;
		List<Data> trainingPoints = trainingData.getData();
		for(int i=0; i < trainingPoints.size(); i++) {
			Data dataPoint = trainingPoints.get(i);
			double actualLabel = dataPoint.getLabelValue();
			double predictedLabel = predictLabel(dataPoint, trainingPoints);
			if(actualLabel * predictedLabel <= 0) {
				alphas[i]+= dataPoint.getLabelValue();
				changeCount++;
			}
		}
		return changeCount;
	}

	private double predictLabel(Data d, List<Data> trainingPoints) throws Exception {
		double result = 0;
		for(int i=0; i< trainingPoints.size(); i++) {
			result+= alphas[i] * kernel.computeValue(d, trainingPoints.get(i), 0, 0, false);
		}
		return Math.signum(result);
	}

	public void test(DataSet trainingData, DataSet testData) throws Exception {
		System.out.println(Arrays.toString(alphas));
		double error = 0;
		for(int i=0; i < testData.getDataSize(); i++) {
			Data d = testData.getData().get(i);
			double actualLabel = d.getLabelValue();
			double predictedLabel = predictLabel(d, trainingData.getData());
			if(actualLabel != predictedLabel) {
				error++;
			}
		}
		System.out.println(error/testData.getDataSize());
	}
}

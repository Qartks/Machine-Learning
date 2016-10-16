package com.ml.hw5;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class AdaBoostImplementation {
	
	DataSet data;
	int rounds;
	boolean optimal;
	List<DTreeStump> models;
	double[] π;
	double[] weights;
	Map<Integer, Set<Double>> map = null;

	public AdaBoostImplementation(int t, boolean b) {
		this.rounds = t;
		this.optimal = b;
		this.π = new double[rounds];
		this.models = new ArrayList<DTreeStump>();
	}

	public void train(DataSet trainData, DataSet testData, boolean generateStats) {
		this.weights = new double[trainData.getDataSize()];
		this.data = trainData;
		Arrays.fill(weights, (double) (1d/trainData.getDataSize()));
		
		if(optimal){
			map = trainData.map;
		}
		int round = 0;
		while(models.size() < rounds){
//			System.out.println("New Round");
			DTreeStump model = weakLearnerModel(trainData, map);
			double epsilon = model.epsilon;
			
			if(epsilon < 0.0005 || epsilon > 0.9995 || (Math.abs(0.5 - epsilon) < 0.01)) {
				continue;
			}
			
			
			models.add(model);
			double alpha = calculateAlpha(epsilon);
			π[round] = alpha;
			updateWeights(alpha, model, trainData);
			
			if(generateStats){
				displayTrainingRoundStats(models.size(), model, trainData,testData, epsilon);
			}
			round++;
		}
	}
	
	private void displayTrainingRoundStats(int round, DTreeStump model, DataSet trainData, DataSet testData, double epsilon) {
		double trainingError = testModel(trainData);
		double testingError = testModel(testData);
		
		System.out.println("Round-> " + round + " :: FeatureIndex-> " + model.featureIndex + " :: Feature Threshold-> " + model.thresholdValue + " :: Round Error-> " + epsilon + " :: TrainError-> " + trainingError + " :: TestError-> " + testingError);
	}

	public double testModel(DataSet dataSet) {
		double error = 0;
		for(Data data : dataSet.getData()){
			double predictedValue = getPredictedLabel(data); 
			double actualValue = data.getLabelValue() == 0 ? -1 : 1;
			if(predictedValue != actualValue){
				error++;
			}
		}
		error /= dataSet.getDataSize();
		return error;
	}

	public double getPredictedLabel(Data data) {
		double val = 0;
		int index = 0;
		for(DTreeStump model : models){
			val += (π[index] * model.getPredicted(data));
			index++;
		}
		if(val >= 0){
			return 1;
		}
		return -1;
	}

	private void updateWeights(double alpha, DTreeStump model, DataSet dataSet) {
		
		double totalWeight = 0;
		for(int i = 0; i < dataSet.getDataSize(); i++){
			Data d = dataSet.getData().get(i);
			double actualLabel = d.getLabelValue() == 0 ? -1 : 1;
			double predictedLabel = model.getPredicted(d);
			
			double oldDataErrorWeight = weights[i];
			double alphaYHt = alpha * actualLabel * predictedLabel;
			double newDataErrorWeight = oldDataErrorWeight * Math.exp(-alphaYHt);
			
			weights[i] = newDataErrorWeight;
			totalWeight+= newDataErrorWeight;
		}
		
		normalizeWeights(totalWeight);
	}

	private void normalizeWeights(double totalWeight) {
		for(int i = 0; i < weights.length; i++){
			weights[i] /= totalWeight;
		}
	}

	private double calculateAlpha(double epsilon) {
		double oneMinusEpsilon = 1 - epsilon;
		return 0.5d * Math.log(oneMinusEpsilon/epsilon);
	}

	private DTreeStump weakLearnerModel(DataSet trainData, Map<Integer, Set<Double>> map) {
		DTreeStump dT = new DTreeStump(trainData, weights);
		if(optimal){
			dT.getOptimalStump(map);
		} else {
			dT.getRandomStump();
		}
		return dT;
	}
}

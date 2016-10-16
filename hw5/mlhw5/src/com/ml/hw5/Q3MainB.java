package com.ml.hw5;

import mltk.core.Instances;
import mltk.core.io.InstancesReader;
import mltk.predictor.Classifier;
import mltk.predictor.Learner.Task;
import mltk.predictor.evaluation.Evaluator;
import mltk.predictor.glm.LassoLearner;

public class Q3MainB {

	public static void main(String[] args) throws Exception  {
		
		Instances trainingInstances = InstancesReader.read("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/totalTrain.txt", 1057);
		Instances testInstances = InstancesReader.read("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spam_polluted/totalTest.txt", 1057);
		Classifier learner = train(trainingInstances);
		System.out.println(Evaluator.evalError(learner, testInstances));
		
	}
	
	public static Classifier train(Instances instances) throws Exception {
		LassoLearner learner = new LassoLearner();
		learner.setVerbose(true);
		learner.setTask(Task.CLASSIFICATION);
		learner.setLambda(0.1);
		learner.setMaxNumIters(100);
		return learner.build(instances);
	}

}

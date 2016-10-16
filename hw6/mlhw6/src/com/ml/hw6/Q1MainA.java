package com.ml.hw6;

import java.io.File;
import java.util.Collections;

import libsvm.LibSVM;
import libsvm.svm_parameter;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

public class Q1MainA {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		
		Normalize.normalizeDataSandS(allData);
		createKFoldFiles(10, allData);
		
		errorWithKFold(10);
		
	}
	
	private static void createKFoldFiles(int k, DataSet dataSet) throws Exception {
		
		int dataPerFold = dataSet.getDataSize()/ k;
		Collections.shuffle(dataSet.getData());
		
		for(int fold=0; fold< k ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getFeatureSize());
			DataSet testData = new DataSet(dataSet.getFeatureSize());
			
			for(int x = 0; x < dataSet.getDataSize(); x++) {
				Data d = dataSet.getData().get(x);
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(d);
					d.setDataSet(testData);
				} else {
					trainingData.addData(d);
					d.setDataSet(trainingData);
				}
			}
			
			DataInputer.writeDataToFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/fold_"+(fold + 1)+"_train_spambase.data", trainingData);
			DataInputer.writeDataToFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/fold_"+(fold + 1)+"_test_spambase.data", testData);
		}
		
	}

	private static void errorWithKFold(int k) throws Exception {
		
		double avgTrainAcc = 0;
		double avgTestAcc = 0;
		float accuracy = 0;
		float correct = 0, wrong = 0;
		for(int fold=0; fold< k ; fold++) {
			
			Dataset data = FileHandler.loadDataset(new File("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/fold_"+(fold + 1)+"_train_spambase.data"), 57, " ");
			
			LibSVM svm = new LibSVM();
			svm_parameter param = new svm_parameter();
			
			// Linear
//			param.C = 100;
//		    param.svm_type = libsvm.svm_parameter.C_SVC;
//		    param.kernel_type = svm_parameter.LINEAR;
//	        param.cache_size = 100.0;
//	        param.coef0=0.0;
//	        param.degree = 1;
//	        param.eps = 0.001;
//	        param.gamma= 0.1;
//	        param.nu = 0.5;
//	        param.probability= 0;
//	        param.shrinking = 1;
			
			// Poly
//			param.C = 20;
//		    param.svm_type = libsvm.svm_parameter.C_SVC;
//		    param.kernel_type = svm_parameter.POLY;
//	        param.cache_size = 100.0;
//	        param.coef0= 1;
//	        param.degree = 2;
//	        param.eps = 0.001;
//	        param.gamma= 0.001;
//	        param.nu = 1;
//	        param.probability= 0;
//	        param.shrinking = 0;
			
			
			// RBF 
			param.C = 10;
		    param.svm_type = libsvm.svm_parameter.C_SVC;
		    param.kernel_type = svm_parameter.RBF;
	        param.cache_size = 100.0;
	        param.coef0=0.0;
	        param.degree = 1;
	        param.eps = 0.05;
	        param.gamma= 0.9;
	        param.nu = 0.5;
	        param.probability= 0;
	        param.shrinking = 1;
	        
	        
//	        param.svm_type = svm_parameter.C_SVC;
//			param.kernel_type = svm_parameter.RBF;
//			param.degree = 3;
//			param.gamma = 0;
//			param.coef0 = 0;
//			param.nu = 0.5;
//			param.cache_size = 40;
//			param.C = 1;
//			param.eps = 1e-3;
//			param.p = 0.1;
//			param.shrinking = 1;
//			param.probability = 0;
//			param.nr_weight = 0;
//			param.weight_label = new int[0];
//			param.weight = new double[0];
	        
	        svm.setParameters(param);
			
//			svm_parameter sp = svm.getParameters();
//			System.out.println(sp.C);
//			System.out.println(sp.svm_type);
//			System.out.println(sp.kernel_type);
//			System.out.println(sp.cache_size);
//			System.out.println(sp.coef0);
//			System.out.println(sp.degree);
//			System.out.println(sp.eps);
//			System.out.println(sp.gamma);
//			System.out.println(sp.nu);
//			System.out.println(sp.probability);
//			System.out.println(sp.shrinking);
	        
	        
	        svm.buildClassifier(data);

	        
	        
	        Dataset traindataForClassification = data;
	        correct = 0;
	        wrong = 0;
	        for (Instance inst : traindataForClassification) {
	            Object predictedClassValue = svm.classify(inst);
	            Object realClassValue = inst.classValue();
	            if (predictedClassValue.equals(realClassValue))
	                correct++;
	            else
	                wrong++;
	        }
	        accuracy = (correct / (correct + wrong));
	        avgTrainAcc += accuracy;
	        System.out.println("Fold "+ (fold + 1) + " Train Accuracy: " + accuracy);
	        
	        Dataset dataForClassification = FileHandler.loadDataset(new File("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/fold_"+(fold + 1)+"_test_spambase.data"), 57, " ");
	        correct = 0;
	        wrong = 0;
	        for (Instance inst : dataForClassification) {
	            Object predictedClassValue = svm.classify(inst);
	            Object realClassValue = inst.classValue();
	            if (predictedClassValue.equals(realClassValue))
	                correct++;
	            else
	                wrong++;
	        }
	        accuracy = (correct / (correct + wrong));
	        avgTestAcc += accuracy;
	        System.out.println("Fold "+ (fold + 1) + " Test Accuracy: " + accuracy);
		}
		
		System.out.println("\n\nAverage Training Accuracy: " + (avgTrainAcc/k));
		System.out.println("Average Testing Accuracy: " + (avgTestAcc/k));
		
	}

}

package com.ml.hw2;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;



public class ConfusionMatrix {

	public static void main(String[] args) throws Exception {

		DataSet spamDataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data",
				"/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names");

		
		Normalize.normalizeDataSandS(spamDataSet);
		
		double avgPerError = 0;
		double avgTrainError = 0;
		int dataPerFold = spamDataSet.dataSize()/ 10;
		
		double[] decisionTreeConMat = new double[4];
		double[] linearRegressConMat = new double[4];
		double[] logicalRegressConMat = new double[4];
		
		
		Collections.shuffle(spamDataSet.getData());
		
		PrintWriter pw = new PrintWriter("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/LRCurve.csv");
		PrintWriter pw2 = new PrintWriter("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/LogCurve.csv");
		

		double areaUnderLR = 0;
		double areaUnderLogR = 0;
		double iter = 0;
		
		for(double thres = 0; thres < 1.0; thres+= 0.002){
			
			double tpr = 0;
			double fpr = 0;
			
			double tpr1 = 0;
			double fpr1 = 0;
			for(int fold=0; fold< 10 ; fold++) {
				
				DataSet trainingData = new DataSet(spamDataSet.getLabelIndex(), spamDataSet.getFeatures());
				DataSet testData = new DataSet(spamDataSet.getLabelIndex(), spamDataSet.getFeatures());
				
				
				for(int x = 0; x < spamDataSet.dataSize(); x++) {
					if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
						testData.addData(spamDataSet.getData().get(x));
					} else {
						trainingData.addData(spamDataSet.getData().get(x));
					}
				}
				
				decisionTreeConMat = getConfusionMatrixForDecisionTree(trainingData, testData, thres);
				linearRegressConMat = getConfusionMatrixForLR(trainingData, testData, thres);
				logicalRegressConMat = getConfusionMatrixForLogR(trainingData, testData, thres);
				
				tpr = linearRegressConMat[0]/(linearRegressConMat[2] + linearRegressConMat[0]);
				fpr = linearRegressConMat[1]/(linearRegressConMat[1] + linearRegressConMat[3]);
//				
				pw.write(tpr +","+ fpr);
				pw.write("\n");
				
				tpr1 = logicalRegressConMat[0]/(logicalRegressConMat[2] + logicalRegressConMat[0]);
				fpr1 = logicalRegressConMat[1]/(logicalRegressConMat[1] + logicalRegressConMat[3]);
				
				pw2.write(tpr1 +","+ fpr1);
				pw2.write("\n");
				
				if(fold == 0){
					break;
				}
			}
			
			
			
		
		}


		System.out.println("Decision Tree: " + Arrays.toString(decisionTreeConMat));
		System.out.println("Linear Regression: " + Arrays.toString(linearRegressConMat));
		System.out.println("Logistic Regression: " + Arrays.toString(logicalRegressConMat));
		
		
		
		pw.close();
		pw2.close();
		
	}

	private static double[] getConfusionMatrixForLR(DataSet trainingData,
			DataSet testData, double thres) throws Exception {
		// TODO Auto-generated method stub
		double[] conMat = new double[4];
		
		PrintWriter pw = new PrintWriter("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/LRValues.csv");
		int trainDataSize = trainingData.getData().size();
		int trainFeatureSize = trainingData.getFeatures().size();
		double[] trainFeatureMatrix = trainingData.getFeatureMatrix();
		double[] trainLabelMatrix = trainingData.getLabelMatrix();
		
		DenseMatrix64F weightMatrix = LRImplementation.train(
				trainDataSize,
				trainFeatureSize,
				trainFeatureMatrix,
				trainLabelMatrix,
				0.1);
		
		int testDataSize = testData.getData().size();
		int testFeatureSize = testData.getFeatures().size();
		double[] testFeatureMatrix = testData.getFeatureMatrix();
		double[] testLabelMatrix = testData.getLabelMatrix();
		
		
		DenseMatrix64F x = new DenseMatrix64F(testDataSize, testFeatureSize, true, testFeatureMatrix);
		DenseMatrix64F y = new DenseMatrix64F(testDataSize, 1, true, testLabelMatrix);
		
		DenseMatrix64F predV = new DenseMatrix64F(testDataSize, 1);
		CommonOps.mult(x, weightMatrix, predV);
		
		for(int i = 0; i< testDataSize; i++){
			double predicted = (predV.get(i, 0) < thres) ? 0 : 1;
			double actual = y.get(i, 0);

//			pw.write(predicted +","+ actual);
//			pw.write("\n");
			
			
			generateConMat(predicted, actual, conMat, pw, testDataSize);
		}
		
		pw.close();
		return conMat;
	}

	private static double[] getConfusionMatrixForLogR(DataSet trainingData,
			DataSet testData, double thres) throws Exception {
		double[] conMat = new double[4];
		
		PrintWriter pw = new PrintWriter("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/LogRValues.csv");
		int trainDataSize = trainingData.getData().size();
		int trainFeatureSize = trainingData.getFeatures().size();
		double[] trainFeatureMatrix = trainingData.getFeatureMatrix();
		double[] trainLabelMatrix = trainingData.getLabelMatrix();
		
		SimpleMatrix weightMatrix= GradDesImplement.getWeights(trainingData, 0.001, true, 0, true, testData);
		
		int testDataSize = testData.getData().size();
		int testFeatureSize = testData.getFeatures().size();
		double[] testFeatureMatrix = testData.getFeatureMatrix();
		double[] testLabelMatrix = testData.getLabelMatrix();
		
//		System.out.println(weightMatrix.toString());
//		System.out.println(testDataSize);
		DenseMatrix64F x = new DenseMatrix64F(testDataSize, testFeatureSize, true, testFeatureMatrix);
		DenseMatrix64F y = new DenseMatrix64F(testDataSize, 1, true, testLabelMatrix);
		DenseMatrix64F w = new DenseMatrix64F(testFeatureSize, 1, true, weightMatrix.getMatrix().getData());
		
		DenseMatrix64F predV = new DenseMatrix64F(testDataSize, 1);
		CommonOps.mult(x, w, predV);
		
		for(int i = 0; i< testDataSize; i++){
			double predicted = (predV.get(i, 0) < thres) ? 0 : 1;
			double actual = y.get(i, 0);

//			pw.write(predicted +","+ actual);
//			pw.write("\n");
			generateConMat(predicted, actual, conMat, pw, testDataSize);
		}
		
		pw.close();
		return conMat;
	}

	private static double[] getConfusionMatrixForDecisionTree(
			DataSet spamDataSet, DataSet testData, double thres) throws Exception {
//
		PrintWriter pw = new PrintWriter("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/decisionTreeValue.csv");
		TreeImplementation decisionTree = new TreeImplementation(7,0.03,39);
		TreeNode root = decisionTree.execute(spamDataSet);
		double[] conMat = new double[4];
		for(Data data : testData.getData()) {
			double predictedValue = decisionTree.predictedValue(root, data);
			double actualValue = data.labelValue();
			
			generateConMat(predictedValue, actualValue, conMat, pw, testData.dataSize());
		}
		
		pw.close();

		return conMat;
		
	}

	private static void generateConMat(double pV,
			double aV, double[] conMat, PrintWriter pw, int testDataSize) {
		
		
		if(pV == 1 && aV == 1){
			conMat[0]++;
		}
		if(pV == 1 && aV == 0){
			conMat[1]++;
		}
		if(pV == 0 && aV == 1){
			conMat[2]++;
		}
		if(pV == 0 && aV == 0){
			conMat[3]++;
		}
		
		pw.write( (conMat[0]/testDataSize) +","+ (conMat[1]/testDataSize));
		pw.write("\n");
		
	}

}

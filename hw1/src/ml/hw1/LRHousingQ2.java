package ml.hw1;

import org.ejml.data.DenseMatrix64F;

public class LRHousingQ2 {

	public static void main(String[] args) throws Exception {
		
		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, true);
		
		DataSet testSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_test.txt", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, true);

		int trainDataSize = dataSet.getData().size();
		int trainFeatureSize = dataSet.getFeatures().size()-1;
		double[] trainFeatureMatrix = dataSet.getFeatureMatrix();
		double[] trainLabelMatrix = dataSet.getLabelMatrix();
		
//		System.out.println(dataSet.getData().size() + "   " + (dataSet.getFeatures().size()-1));
//		System.out.println(dataSet.getFeatureMatrix().length + " " + dataSet.getFeatureMatrix()[0].length);
		
		DenseMatrix64F weightMatrix = LRImplementation.train(
				trainDataSize,
				trainFeatureSize,
				trainFeatureMatrix,
				trainLabelMatrix);

		double trainingError = LRImplementation.calculateError(weightMatrix,
				trainDataSize,
				trainFeatureSize,
				trainFeatureMatrix,
				trainLabelMatrix);
		
		System.out.println("Error on Training Set -> " + trainingError);
		
		int testDataSize = testSet.getData().size();
		int testFeatureSize = testSet.getFeatures().size()-1;
		double[] testFeatureMatrix = testSet.getFeatureMatrix();
		double[] testLabelMatrix = testSet.getLabelMatrix();

		double testingError = LRImplementation.calculateError(weightMatrix,
				testDataSize,
				testFeatureSize,
				testFeatureMatrix,
				testLabelMatrix);
		
		System.out.println("Error on Testing Set -> " + testingError);

	}

}

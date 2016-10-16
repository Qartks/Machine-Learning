package ml.hw1;

import org.ejml.data.DenseMatrix64F;


public class LRSpamQ2 {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names", false, false);
		
//		System.out.println(dataSet.getData().size() + "   " + (dataSet.getFeatures().size()-1));
//		System.out.println(dataSet.getFeatureMatrix().length + " " + dataSet.getFeatureMatrix()[0].length);
//		DenseMatrix64F weightMatrix = LRImplementation.train(
//				dataSet.getData().size(),
//				dataSet.getFeatures().size()-1,
//				dataSet.getFeatureMatrix(),
//				dataSet.getLabelMatrix());
		
		errorWithKFold(10, dataSet);
	}
	
private static void errorWithKFold(int k, DataSet dataSet) throws Exception {
		
		double avgPerError = 0;
		int dataPerFold = dataSet.dataSize()/ k;
		
		for(int fold=0; fold< k ; fold++) {
			
			DataSet trainingData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			DataSet testData = new DataSet(dataSet.getLabelIndex(), dataSet.getFeatures());
			
			
			for(int x = 0; x < dataSet.dataSize(); x++) {
				if(x >= fold * dataPerFold && x < (fold+1)*dataPerFold) {
					testData.addData(dataSet.getData().get(x));
				} else {
					trainingData.addData(dataSet.getData().get(x));
				}
			}
			
			DenseMatrix64F weightMatrix = LRImplementation.train(
					trainingData.getData().size(),
					trainingData.getFeatures().size()-1,
					trainingData.getFeatureMatrix(),
					trainingData.getLabelMatrix());
			
			double error = LRImplementation.calculateError(weightMatrix,
					dataSet.getData().size(),
					dataSet.getFeatures().size()-1,
					dataSet.getFeatureMatrix(),
					dataSet.getLabelMatrix());
			
			avgPerError += (error);
			System.out.println("Error in fold "+ (fold+1) + "-> "+error);
			
		}
		System.out.println("Total average error: " + avgPerError/ k);
		
	}


}

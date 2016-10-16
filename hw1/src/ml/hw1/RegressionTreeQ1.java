package ml.hw1;

public class RegressionTreeQ1 {
	
	public static void main(String[] args) throws Exception {
		
		DataSet trainingData = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_train.txt",  "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, true);
		DataSet testData = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_test.txt", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/housing_features.txt", false, false);

		TreeImplementation regressionTree = new TreeImplementation(5, 15, 39);
		TreeNode root = regressionTree.execute(trainingData);
		errorResults(trainingData, regressionTree, root);
		errorResults(testData, regressionTree, root);
	}
	
	public static void errorResults(DataSet testData, TreeImplementation regressionTree, TreeNode root) throws Exception {
		double diffSum = 0;
		
		for(Data data : testData.getData()){
			double perdictedValue = regressionTree.predictedValue(root,data);
			diffSum += Math.pow((data.labelValue() - perdictedValue), 2);
		}
		
		double mse = diffSum/testData.getData().size();
		
		System.out.println("Mean Square Error:" + mse);
	}

}

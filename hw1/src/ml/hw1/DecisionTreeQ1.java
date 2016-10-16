package ml.hw1;

public class DecisionTreeQ1 {

	public static void main(String[] args) throws Exception {
		TreeImplementation decisionTree = new TreeImplementation(7,0.03,39);
		DataSet dataSet = DataInputer.getDataFromFile("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data", "/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.names", false, false);
		
		
		errorWithKFold(10, dataSet, decisionTree);
		
		
	}

	private static void errorWithKFold(int k, DataSet dataSet, TreeImplementation decisionTree) throws Exception {
		
		double avgError = 0;
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
			
			TreeNode root = decisionTree.execute(trainingData);
			int error = 0;
			double errorOnFold = 0;
			for(Data data : testData.getData()) {
				if(data.labelValue() != decisionTree.predictedValue(root, data)) {
					error++;
				}
			}
			errorOnFold = ((double)error)/dataPerFold*100;
			avgPerError += errorOnFold;
			System.out.println("Error on Fold "+ (fold+1) + " -> " + errorOnFold);
		}
		System.out.println("Average Error Percentage :"+ avgPerError/ k);
	
	}

}

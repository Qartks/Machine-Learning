package ml.hw1;

public class NodeStats {
	
	private int[] labelsPerClass; // Used for categorical data
	private int totalData;
	private double totalValue;
	
	public NodeStats() {
		totalData = 0;
		totalValue = 0;
	}
	
	public NodeStats(int classNum) {
		labelsPerClass = new int[classNum];
		totalData = 0;
		totalValue = 0;
	}
	
	
	public void add(Data data) throws Exception {
		int labelIndex = data.labelIndex();
		if(data.getFeature(labelIndex).isNominal()) {
			labelsPerClass[(int) data.labelValue()]++;
		} else {
			totalValue += data.labelValue();
		}
		totalData++;
	}
	
	
	public int predictClass() {
		int predictedClass = -1;
		int maxLabelCount = -1;
		for(int i=0; i< labelsPerClass.length; i++) {
			if(labelsPerClass[i] > maxLabelCount) {
				predictedClass = i;
				maxLabelCount = labelsPerClass[i];
			}
		}
		return predictedClass;
	}
	
	public double predictMean() {
		return totalValue/totalData;
	}
	
	public int[] getLabelPerClass() {
		return labelsPerClass;
	}
	
	public int getTotalData() {
		return totalData;
	}

	public double getTotalValue() {
		return totalValue;
	}

	public void setTotalValue(double totalValue) {
		this.totalValue = totalValue;
	}

}

package ml.hw1;

import java.util.List;

public class Data {
	
	private double[] featureValues;
	private DataSet dataset;
	
	public Data(Data data) {
		featureValues = data.getFeatureValues();
		dataset = data.getDataSet();
	}

	public Data(double[] values) {
		this.featureValues = values;
		this.dataset = null;
	}
	
	public double[] getFeatureValues() {
		return featureValues;
	}
	

	public double getFeatureValue(int index) {
		return featureValues[index];
	}
	

	public DataSet getDataSet() {
		return dataset;
	}
	

	public void setDataSet(DataSet dataset) {
		this.dataset = dataset;
	}
	

	public int labelIndex() throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getLabelIndex();
	}
	

	public double labelValue() throws Exception {
		return featureValues[labelIndex()];
	}
	

	public Feature getFeature(int index) throws Exception {
		if(dataset == null) {
			throw new Exception("DataSet is null");
		}
		return dataset.getFeature(index);
	}

	public List<Feature> getFeatures() throws Exception {
		if(dataset == null) {
			throw new Exception("Dataset is null");
		}
		return dataset.getFeatures();
	}
	
	@Override
	public String toString() {
		StringBuilder builder = new  StringBuilder();
		for(double value : featureValues) {
			builder.append(String.valueOf(value)+" ");
		}
		return builder.toString().trim();
	}
}

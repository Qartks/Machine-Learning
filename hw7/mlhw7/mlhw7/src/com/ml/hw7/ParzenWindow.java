package com.ml.hw7;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class ParzenWindow {
	
	Kernel kernel;
	DataSet trainData;
	
	public ParzenWindow(Kernel k, DataSet trainData) {
		this.kernel = k;
		this.trainData = trainData;
	}

	public double execute(DataSet testData) throws Exception {
		double error = 0;
		for(Data d : testData.getData()){
			double actualLabel = d.labelValue;
			double predictedLabel = predictedValue(d);
			if(actualLabel != predictedLabel){
				error++;
			}
		}
		return error/testData.getDataSize();
	}

	private double predictedValue(Data d) throws Exception {
		List<KNNPoint> neighbours = getNeighbours(d);
		return getMaxLabel(neighbours);
	}
	
	private double getMaxLabel(List<KNNPoint> neighbours) {
		
		Map<Double, Double> labelNeighboursCount = new HashMap<Double, Double>();
		for(KNNPoint neighbour : neighbours) {
			Data data = neighbour.p;
			if(labelNeighboursCount.containsKey(data.getLabelValue())) {
				labelNeighboursCount.put(data.getLabelValue(), labelNeighboursCount.get(data.getLabelValue()) + (neighbour.similarity));
			} else {
				labelNeighboursCount.put(data.getLabelValue(), neighbour.similarity);
			}
		}
		
		double maxLabelContribution = Double.NEGATIVE_INFINITY;
		double maxLabel = Double.NaN;
		
		for(Entry<Double, Double> entry : labelNeighboursCount.entrySet()) {
			double label = entry.getKey();
			double labelCount = entry.getValue();
			if(labelCount > maxLabelContribution) {
				maxLabelContribution = labelCount;
				maxLabel = label;
			}
		}
		return maxLabel;
	}

	private List<KNNPoint> getNeighbours(Data d) throws Exception {
		List<KNNPoint> all = getAllNeighbors(d);
		return all;
	}

	private List<KNNPoint> getAllNeighbors(Data d) throws Exception {
		List<KNNPoint> list = new ArrayList<KNNPoint>();
		for(int i = 0 ; i < trainData.getDataSize(); i++){
			Data di = trainData.getData().get(i);
			double ker = kernel.computeValue(d, di, i, i, false);
			KNNPoint kp = new KNNPoint(di, ker);
			list.add(kp);
		}
		return list;
	}
}

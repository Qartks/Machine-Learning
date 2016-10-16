package com.ml.hw7;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class KNNImplementation {
	
	int k;
	double[][] cache;
	Kernel kernel;
	DataSet dataSet;
	double range;
	boolean useWindow;
	
	public KNNImplementation(int i, double range, boolean useWindow, DataSet dataSet, Kernel iKer) {
		this.k = i;
		this.dataSet = dataSet;
		kernel = iKer;
		this.range = range;
		this.useWindow = useWindow;
	}
	
	public double execute(DataSet ds) throws Exception {
		double error = 0;
		for(Data d : ds.getData()){
			double actualLabel = d.labelValue;
			double predictedLabel = getPredictedValue(d);
			if(actualLabel != predictedLabel){
				error++;
			}
		}
//		System.out.println("Error -> " + (error/ds.getDataSize()));
		return (error/ds.getDataSize());
	}

	private double getPredictedValue(Data d) throws Exception {
		List<KNNPoint> neighbours = getNeighbours(d);
		return evalulateNeighbours(neighbours);
	}

	private double evalulateNeighbours(List<KNNPoint> neighbours) {
		Map<Double, Integer> labelNeighboursCount = new HashMap<Double, Integer>();
		for(KNNPoint neighbour : neighbours) {
			Data data = neighbour.p;
			if(labelNeighboursCount.containsKey(data.labelValue)) {
				labelNeighboursCount.put(data.labelValue, (int)labelNeighboursCount.get(data.labelValue)+1);
			} else {
				labelNeighboursCount.put(data.labelValue, 1);
			}
		}
		
		double maxLabelCount = Double.NEGATIVE_INFINITY;
		double maxLabel = Double.NaN;
		
		for(Entry<Double, Integer> entry : labelNeighboursCount.entrySet()) {
			double label = entry.getKey();
			int labelCount = entry.getValue();
			if(labelCount > maxLabelCount) {
				maxLabelCount = labelCount;
				maxLabel = label;
			}
		}
		return maxLabel;
	}

	private List<KNNPoint> getNeighbours(Data d) throws Exception {
		List<KNNPoint> all = getAllNeighbors(d);
		if(useWindow){
			return getNeighboursWithinRange(all);
		}
		return getKNearestNeighbours(all, k);
	}
	
	private List<KNNPoint> getNeighboursWithinRange(List<KNNPoint> all) {
		List<KNNPoint> neighboursInRange = new ArrayList<KNNPoint>();
		for(KNNPoint neighbour : all) {
			if(neighbour.similarity > -range) {
				neighboursInRange.add(neighbour);
			}
		}
		return neighboursInRange;
	}

	private List<KNNPoint> getKNearestNeighbours(List<KNNPoint> all, int k) {
		List<KNNPoint> nearestNeighbours = new ArrayList<KNNPoint>();
		for(int i=0; i < k; i++) {
			nearestNeighbours.add(all.get(i));
		}
		return nearestNeighbours;
	}

	private List<KNNPoint> getAllNeighbors(Data d) throws Exception {
		List<KNNPoint> neigh = new ArrayList<KNNPoint>();
		for(int i = 0 ; i < dataSet.getDataSize(); i++){
			Data di = dataSet.getData().get(i);
			double ker = kernel.computeValue(d, di, i, i, false);
			KNNPoint kp = new KNNPoint(di, ker);
			neigh.add(kp);
		}
		Collections.sort(neigh, new Comparator<KNNPoint>() {
			@Override
			public int compare(KNNPoint o1, KNNPoint o2) {
				double sim1 = o1.similarity;
				double sim2 = o2.similarity;
				return -Double.compare(sim1, sim2);
			}
		});
		
		return neigh;
	}

}

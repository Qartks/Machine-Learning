package com.ml.hw7;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Relief {
	
	double[] weights;
	DataSet dataSet;
	Kernel kernel;

	public Relief(int featureSize, DataSet trainingData, Kernel kernel) {
		this.weights = new double[featureSize];
		this.dataSet = trainingData;
		this.kernel = kernel;
	}

	public double[] generateImportantFeatures(DataSet trainingData) {
		for(int i = 0; i < trainingData.getFeatureSize(); i++){
			for(int j = 0 ; j < trainingData.getDataSize(); j++){
				Data dj = trainingData.getData().get(j);

				double closestSame = Double.POSITIVE_INFINITY;
				double closestOppO = Double.POSITIVE_INFINITY;
				for(int k = 0 ; k < trainingData.getDataSize(); k++){
					if( k == i){
						continue;
					}
					Data dk = trainingData.getData().get(k);
					if(dk.labelValue == dj.labelValue){
						double diff = getDifference(dj, dk); 
						if(diff < closestSame){
							closestSame = diff;
						}
					} else {
						double diff = getDifference(dj, dk); 
						if(diff < closestOppO){
							closestOppO = diff;
						}
					}
				}

				weights[i] = weights[i] - closestSame + closestOppO; 
			}
		}
		System.out.println(Arrays.toString(weights));
		return weights;
	}
	
	
	public double[] getImpFeatures() throws Exception{
		for(Data d : dataSet.getData()){
			for(int j = 0; j < dataSet.getFeatureSize(); j++){
				Data same = null;
				Data oppo = null;
				
				List<KNNPoint> list = getAllNeighbors(d);
				double labelD = d.labelValue;
				
				for(KNNPoint knnp : list){
					if(same == null && labelD == knnp.getLabel()){
						same = knnp.p;
					}
					if(oppo == null && labelD != knnp.getLabel()){
						oppo = knnp.p;
					}
					if(same != null && oppo != null){
						break;
					}
				}
				
				updateWeights(j, same, oppo, d);
			}
		}
		return weights;
	}

	private void updateWeights(int j, Data same, Data oppo, Data data) {
		weights[j] = weights[j] - (data.getFeatureValueAtIndex(j) - same.getFeatureValueAtIndex(j)) + (data.getFeatureValueAtIndex(j) - oppo.getFeatureValueAtIndex(j));
	}

	private double getDifference(Data dj, Data dk) {
		double sum = 0;
		for(int i = 0 ; i < dj.getFeatureValues().length; i++){
			sum += (dj.getFeatureValueAtIndex(i) - dk.getFeatureValueAtIndex(i));
		}
		return Math.pow(sum,2);
	}
	
	private List<KNNPoint> getNeighbours(Data d) throws Exception {
		List<KNNPoint> all = getAllNeighbors(d);
		return getKNearestNeighbours(all, 10);
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

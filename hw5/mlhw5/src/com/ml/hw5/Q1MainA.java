package com.ml.hw5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class Q1MainA {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		AdaBoostImplementation ada = new AdaBoostImplementation(300, true);
		
		allData.computeFeatureStats();
		allData.computeOptimalThreVal();
		ada.train(allData, allData, true); 
		
		calculateFeatureRanks(ada.models, ada.π, allData);
		
//		for(int i = 1; i<=1057; i++){
//			System.out.print("Feature-" + i + " ");
//		}

	}
	

	private static void calculateFeatureRanks(List<DTreeStump> models, double[] π, DataSet dataSet) {
		List<RankFeature> list = new ArrayList<RankFeature>();
		
		for(DTreeStump dt : models){
			RankFeature r = new RankFeature(dt.featureIndex, dt.thresholdValue, π);
			if(list.contains(r)){
				RankFeature rr = list.get(list.indexOf(r));
				rr.incrementCount(models.indexOf(dt));
			} else {
				list.add(r);
				r.incrementCount(models.indexOf(dt));
			}
		}
		
		double margin = calculateTotoalMarginOnData(models, π, dataSet);
		
		for(RankFeature r : list){
			r.calculateMargin(models, dataSet, margin);
		}
		
		Collections.sort(list, new Comparator<RankFeature>(){

			@Override
			public int compare(RankFeature o1, RankFeature o2) {
				return -Double.compare(o1.margin, o2.margin);
			}
			
		});
		
		int count = 0;
		for(RankFeature r : list){
			if(count < 20){
				System.out.println(r);
			} else {
				break;
			}
			count++;
		}
	}


	private static double calculateTotoalMarginOnData(List<DTreeStump> models, double[] π, DataSet dataSet) {
		double avgMar = 0;
		for(int i = 0; i< dataSet.getDataSize(); i++){
			double m = 0;
			double sum = 0;
			Data d = dataSet.getData().get(i);
			for(int ind = 0; ind < models.size(); ind++){
				DTreeStump model = models.get(ind);
				m += (model.getPredicted(d) * π[ind]);
			}
			double yi = d.getLabelValue() == 0 ? -1 : 1;
			double avg = m * yi; 
			avgMar += avg;
		}
		return avgMar;
	}

}

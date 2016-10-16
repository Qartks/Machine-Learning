package com.ml.hw3;

import java.util.List;


public class EMImplementation {
	
	int maxIteration;
	EMModel model;
	
	public EMImplementation(int n, DataSet dataSet, List<GaussianModel> list, double[] π, int iter) {
		this.model = new EMModel(n, dataSet, π);
		this.model.setmList(list);
		this.maxIteration = iter;
//		System.out.println(M +" " + N);
//		System.out.println(Arrays.toString(π));
	}
	
	public EMModel getEmModel(){
		return model;
	}
	public void setMaxIteration(int maxIteration) {
		this.maxIteration = maxIteration;
	}

	public void run() {
		for( int step=0; step <= maxIteration; step++ ) {
			model.mStep();
			model.eStep();
		}
	}

	

}

package com.ml.hw2;

import java.util.ArrayList;
import java.util.Arrays;

import org.ejml.simple.SimpleMatrix;

public class PerceptronImplementation {
	
	
	// Initialize w ( 0 ... d )
	// do 
	//			M <- get set of misclassified points
	//			For each data point in M
	//				update-rule => w <- w + ( lambda * xT )
	// while (M is not empty)
	
	public static void learn(SimpleMatrix w, SimpleMatrix x, SimpleMatrix y, double lambda){
		
		ArrayList<Integer> m = getMisclassified(w, x, y);
		int iter = 1;
		while(!m.isEmpty()){
//		while(iter < 2){
			System.out.println("Iteration " + iter +" , total_mistake " + m.size());
//			System.out.println(m.toString());
			for(int i : m){
				SimpleMatrix row = x.extractVector(true, i);
//				System.out.println(row.toString());
				SimpleMatrix b = row.transpose();
				if(y.get(i, 0) > 0){
					w = w.plus(lambda,b);
				} else {
					w = w.plus(-lambda,b);
				}
			}
			iter++;
			m = getMisclassified(w, x, y);
		}
		System.out.println("Iteration " + iter +" , total_mistake " + m.size());
		
		System.out.println("Classifier weights: " + Arrays.toString((w.getMatrix().getData())));
		
		double w0 = w.getMatrix().getData()[0];
		double[] weightsFinal = new double[4];
		
		for(int i = 0; i< 4; i++){
			if(w0 != 0)
				weightsFinal[i] = w.getMatrix().getData()[i+1] / -w0;
		}
		
		System.out.println();
		System.out.println("Normalized with threshold: " + Arrays.toString(weightsFinal));
		
	}
	
	
	

	private static ArrayList<Integer> getMisclassified(SimpleMatrix w,
			SimpleMatrix x, SimpleMatrix y) {

		ArrayList<Integer> m = new ArrayList<Integer>();
		
		for(int i = 0; i< 1000; i++){
			SimpleMatrix row = x.extractVector(true, i);
			double classify = row.mult(w).getMatrix().getData()[0];
//			System.out.println(classify);
			double yLabel = y.get(i, 0);
			if(classify >= 0 && yLabel < 0){
				m.add(i);
			} else if(classify < 0 && yLabel > 0){
				m.add(i);
			}
			
//			if(classify * yLabel <= 0){
//				m.add(i);
//			}
		}
		
		return m;
		
	}
}

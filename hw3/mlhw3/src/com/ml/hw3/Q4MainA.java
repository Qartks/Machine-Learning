package com.ml.hw3;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public class Q4MainA {

	public static void main(String[] args) throws Exception{
		
		DataInputer.noOFFeatures = 2;
		DataSet allData = DataInputer.getEMData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/HW3/2gaussian.txt");
		
		double[] mean_1 = {Math.random(),Math.random()}; 
//		double[] mean_1 = {1,2}; 
		
//		double[][] cov_1 = {{Math.random(),Math.random()},{Math.random(),Math.random()}};
		double[][] cov_1 = {{1,0},{0,1}};
		
		double[]mean_2 ={Math.random(),Math.random()}; 
//		double[]mean_2 ={1,2}; 
		
//		double[][] cov_2 = {{Math.random(),Math.random()},{Math.random(),Math.random()}};
		double[][]cov_2 = {{1,0},{0,1}};
		
		double[] π = {Math.random(), Math.random()};
//		double[] π = {0.6, 0.4};	
		
		GaussianModel m1 = new GaussianModel();
		m1.setMean(new SimpleMatrix(2, 1, true, mean_1));
		m1.setCovariance(new SimpleMatrix(cov_1));
		
		GaussianModel m2 = new GaussianModel();
		m2.setMean(new SimpleMatrix(2, 1, true, mean_2));
		m2.setCovariance(new SimpleMatrix(cov_2));
		
		List<GaussianModel> list = new ArrayList<GaussianModel>();
		list.add(m1);
		list.add(m2);
		
		// Executing
		EMImplementation em = new EMImplementation(2, allData, list, π, 100);
		em.run();
		
		// Displaying results
		EMModel emm = em.getEmModel();
		System.out.println(emm.toString());
	}

}

package com.ml.hw3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ejml.alg.dense.linsol.svd.SolvePseudoInverseSvd;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;

public class EMModel {
	
	int M;
	int N;
	double[] π;
	double[][] featVal;
	double[][] z;
	int d;
	// mixture coefficient => 1xM - Probability of each model   [ 1/3 , 2/3 ]
	// Z => NxM - boolean matrix indicating if which model is currently being selected (1), and not (0). 
	
	List<GaussianModel> mList; 
	
	
	public EMModel( int m, DataSet dataSet, double[] iπ) {
		this.M = m;
		this.N = dataSet.getDataSize();
		this.mList = new ArrayList<GaussianModel>(M);
		this.z = new double[N][M];
		this.featVal = dataSet.getFeatureArray();
		this.d = dataSet.getFeatureSize();
		this.π = iπ;
		initilize();
	}

	private void initilize() {
		for(int i = 0; i < N; i++){
			int index = (int) ((Math.random() * 157) % this.M);
//			System.out.println(i);
//			System.out.println(index);
			z[i][index] = 1;
		}
	}

	public List<GaussianModel> getmList() {
		return mList;
	}

	public void setmList(List<GaussianModel> mList) {
		this.mList = mList;
	}

	void setπ(double[] n, double size){
		for(int i = 0; i< n.length; i++){
			π[i] = n[i] / size;
		}
	}

	public void eStep() {
		for(int i = 0; i<N; i++){
			double sum = 0;
			for(int m = 0; m < M; m++){
				GaussianModel gm = mList.get(m);
				double val = calculateProbability(gm, featVal[i]);
				z[i][m] = val * π[m];
				sum += (val * π[m]);
			}
			
			if(sum == 0){
				System.out.println("Hi");
			}
			for(int m = 0; m < M; m++){
				z[i][m] /= sum;
			}
			
//			System.out.println(Arrays.toString(z[i]));
		}
	}

	private double calculateProbability(GaussianModel gm, double[] row) {
		
		SimpleMatrix x = new SimpleMatrix(row.length, 1, true, row);
		SimpleMatrix µ = gm.getΜean();
		SimpleMatrix coVar = gm.getCovariance();
		
		double pi2 = Math.pow((2 * Math.PI), -d/2);
		double deter = Math.pow(coVar.determinant(), - 0.5);
		SimpleMatrix xMinusµ = x.minus(µ);
		
		if(coVar.determinant() < 0){
			System.out.println("Hi");
		}
		SolvePseudoInverseSvd sd = new SolvePseudoInverseSvd();
		DenseMatrix64F mm = new DenseMatrix64F(d, d);
		DenseMatrix64F coVarInv = new DenseMatrix64F(coVar.numRows(), coVar.numCols(), true, coVar.getMatrix().getData());
		sd.setA(coVar.getMatrix());
		sd.invert(mm);
		
		SimpleMatrix xx = new SimpleMatrix(mm.numRows, mm.numCols, true, mm.getData());

		SimpleMatrix exp = xMinusµ.transpose().mult(xx).mult(xMinusµ);
//		System.out.println(exp.toString());
		double det = exp.determinant();
		double expVal = Math.exp(-0.5 * det);
		return pi2 * deter * expVal;
	}

	public void mStep() {
		double sumZ = 0;
		for(int m = 0; m < M; m++){
			sumZ = 0;
			GaussianModel gm = mList.get(m);
			SimpleMatrix co = new SimpleMatrix(d, d);
			SimpleMatrix mean = new SimpleMatrix(d, 1);
			for(int i = 0; i < N; i++){
				sumZ += z[i][m];
				SimpleMatrix tempMat = calculateCovarValue(featVal[i], gm, z[i][m]);
				SimpleMatrix tempMat2 = calculateMeanValue(featVal[i], z[i][m]);
//				if(i == 0){
//					co = tempMat;
//					mean = tempMat2;
//				} else {
					co = co.plus(tempMat);
					mean = mean.plus(tempMat2);
//				}
			}
			co = co.divide(sumZ);
			mean = mean.divide(sumZ);
			π[m] = sumZ/N;
//			System.out.println(co.toString());
			gm.setCovariance(co);
			gm.setMean(mean);
		}
	}
	
	
	private SimpleMatrix calculateMeanValue(double[] row, double z) {
		SimpleMatrix x = new SimpleMatrix(row.length, 1, true, row);
		return x.scale(z);
	}

	private SimpleMatrix calculateCovarValue(double[] row, GaussianModel gm, double z) {
		SimpleMatrix x = new SimpleMatrix(row.length, 1, true, row);
		SimpleMatrix µ = gm.getΜean();
		SimpleMatrix xMinusµ = x.minus(µ);
		SimpleMatrix xMinusµMult = xMinusµ.mult(xMinusµ.transpose());
//		System.out.println(xMinusµMult);
		return xMinusµMult.scale(z);
	}

	public String toString(){
		StringBuilder sb = new StringBuilder();
		StringBuilder sb2 = new StringBuilder();
		
		for(int i = 0; i< mList.size(); i++){
			GaussianModel g = mList.get(i);
			sb.append("\nModel " + (i+1) + " => \n");
			sb.append(g.toString());
			sb.append("\n");
		}
		sb.append("\n\n");
		sb.append("Mixture Coefficients =>");
		sb.append(Arrays.toString(π));
		sb.append("\n");
		sb.append("Count of Data Points =>");
		for(double p : π){
			sb2.append(p * N);
			sb2.append(" ");
		}
		sb.append(sb2.toString().trim());
		
		return sb.toString().trim();
	}

}

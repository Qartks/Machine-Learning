package com.ml.hw5;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SpecializedOps;

public class LRImplementation {
	
	public static double calculateError(DenseMatrix64F weightMatrix, int dataSize, int featureSize,
			double[] featureMatrix, double[] labelMatrix) {
		
		DenseMatrix64F x = new DenseMatrix64F(dataSize, featureSize, true, featureMatrix);
		DenseMatrix64F y = new DenseMatrix64F(dataSize, 1, true, labelMatrix);
		
		DenseMatrix64F weightMultX = new DenseMatrix64F(dataSize, 1);
		CommonOps.mult(x, weightMatrix, weightMultX);
		
		DenseMatrix64F errorMatrix = new DenseMatrix64F(dataSize, 1);
		CommonOps.subtract(weightMultX, y, errorMatrix);
		
		DenseMatrix64F errorSquare = new DenseMatrix64F(1, 1);
		CommonOps.multTransA(errorMatrix, errorMatrix, errorSquare);
		
		return CommonOps.det(errorSquare)/dataSize;
	}

	public static DenseMatrix64F train(int dataSize, int featureSize,
			double[] featureMatrix, double[] labelMatrix, double lambda) {
		
		DenseMatrix64F x = new DenseMatrix64F(dataSize, featureSize, true, featureMatrix);
		DenseMatrix64F y = new DenseMatrix64F(dataSize, 1, true, labelMatrix);
		
		DenseMatrix64F xTranspose = new DenseMatrix64F(featureSize, dataSize);
		CommonOps.transpose(x, xTranspose);
		
		DenseMatrix64F xMultXTranspose = new DenseMatrix64F(featureSize, featureSize);
		CommonOps.mult(xTranspose, x, xMultXTranspose);
		
		DenseMatrix64F xPlusLambda = new DenseMatrix64F(featureSize, featureSize);
		SpecializedOps.addIdentity(xMultXTranspose, xPlusLambda, lambda);
		
		DenseMatrix64F xInverse = new DenseMatrix64F(featureSize, featureSize);
		CommonOps.invert(xPlusLambda, xInverse);
		
		DenseMatrix64F xInverseMultXTranspose = new DenseMatrix64F(featureSize, dataSize);
		CommonOps.multAdd(xInverse, xTranspose, xInverseMultXTranspose);
		
		DenseMatrix64F weights = new DenseMatrix64F(featureSize, 1);
		CommonOps.multAdd(xInverseMultXTranspose, y, weights);
		
		return weights;
	}

}

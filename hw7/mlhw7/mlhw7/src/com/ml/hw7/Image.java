package com.ml.hw7;

import java.util.Arrays;


public class Image {

	int[][] image;
	byte label;
	int numRows;
	int numCols;
	int id;
	
	public Image(int r, int c, int id) {
		this.image = new int[r][c];
		this.numRows = r;
		this.numCols = c;
		this.id = id;
	}
	
	public void setImage(int[][] m){
		this.image = m;
	}
	
	public void setLabel(byte l){
		this.label = l;
	}
	
	public int getImagePixel(int i, int j){
		return image[i][j];
	}

	@Override
	public String toString() {
		for (int colIdx = 0; colIdx < numCols; colIdx++) {
			for (int rowIdx = 0; rowIdx < numRows; rowIdx++) {
				System.out.print(image[colIdx][rowIdx]);
			}
			System.out.println();
		}
		return "Image";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(image);
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Image other = (Image) obj;
		if (!Arrays.deepEquals(image, other.image))
			return false;
		return true;
	}

	public double getLabel() {
		return (double) label;
	}
	
	
}

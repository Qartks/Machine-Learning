package com.ml.hw6;

public class myPair {
	
	int i1;
	int i2;
	
	public myPair(int x, int y) {
		i1 = x;
		i2 = y;
	}

	@Override
	public String toString() {
		return "myPair [i1=" + i1 + ", i2=" + i2 + "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		final int prime2 = 13;
		int result = 1;
		result = prime * result + Math.abs(i1 - i2);
//		result = prime2 * result + i1 + i2;
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
		myPair other = (myPair) obj;
		if (i1 != other.i1 && i2 != other.i2) {
			if(i1 != other.i2 && i2 != other.i1){
				return false;
			}
		}
		
		return true;
	}

}

package com.ml.hw6;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

public class Rectangle {

	Point topLeft;
	Point bottomRight;
	Point topRight;
	Point bottomLeft;
	Point halfLeft;
	Point halfRight;
	Point halfTop;
	Point halfBottom;
	Map<Image, BlackCount> mapBlack;

	public Rectangle(Point p1, Point p2) {
		this.topLeft = p1;
		this.bottomRight = p2;
		
		this.topRight = new Point(bottomRight.x - this.getHeight(), bottomRight.y);
		this.bottomLeft =  new Point(topLeft.x + this.getHeight(), topLeft.y);
		
		this.halfLeft =  new Point(topLeft.x + this.getHeight()/2 , topLeft.y);
		this.halfRight = new Point((topRight.x + bottomRight.x)/2, topRight.y); 
		
		this.halfTop = new Point(topLeft.x, topLeft.y + this.getWidth()/2);
		this.halfBottom = new Point(bottomLeft.x , bottomLeft.y + this.getWidth()/2);
		
		this.mapBlack = new HashMap<Image, BlackCount>();
	}

	public Rectangle(int x1, int y1, int x2, int y2) {
		this.topLeft = new Point(x1, y1);
		this.bottomRight = new Point(x2, y2);
		
		this.topRight = new Point(bottomRight.x - this.getWidth(), bottomRight.y);
		this.bottomLeft =  new Point(topLeft.x + this.getHeight(), topLeft.y);
		
		this.halfLeft =  new Point(topLeft.x + this.getWidth()/2 , topLeft.y);
		this.halfRight = new Point((topRight.x+bottomRight.x)/2, topRight.y); 
		
		this.halfTop = new Point(topLeft.x, topLeft.y + this.getWidth()/2);
		this.halfBottom = new Point(bottomLeft.x , bottomLeft.y + this.getWidth()/2);
		
		this.mapBlack = new HashMap<Image, BlackCount>();
	}
	
	public void calculateBlackCount(Map<Image, Map<Point, Integer>> cache){
		for(Entry e : cache.entrySet()){
			Map<Point, Integer> map = (Map<Point, Integer>) e.getValue();
			Image img = (Image) e.getKey();
			BlackCount b = new BlackCount();
//			black(rectangle ABCD) = black(OTYD) - black(OTXB) - black(OZYC) + black(OZXA)
			
			int blackCount =  countInsidePoints(topLeft, topRight, bottomLeft, bottomRight, map);
			
			int verticalTopCount =  countInsidePoints(topLeft, topRight, halfLeft, halfRight, map);
			int verticalBottomCount =  countInsidePoints(halfLeft, halfRight, bottomLeft, bottomRight, map);
			
			int horizontalLeftCount =  countInsidePoints(topLeft, halfTop, bottomLeft, halfBottom, map);
			int horizontalRightCount =  countInsidePoints(halfTop, topRight, halfBottom, bottomRight, map);
			
			b.setTotalBlackCount(blackCount);
			b.setVecticalCount(verticalTopCount - verticalBottomCount);
			b.setHorizontalCount(horizontalLeftCount - horizontalRightCount);
			mapBlack.put(img, b);
//			System.out.println("MapBlack ->" + mapBlack.size());
		}
	}
	
	
	private int countInsidePoints(Point itopLeft, Point itopRight, Point ibottomLeft, Point ibottomRight, Map<Point, Integer> map) {

		int blackTopLeft = map.get(itopLeft);
		int blackTopRight = map.get(itopRight);
		int blackBottomLeft = map.get(ibottomLeft);
		int blackBottomRight = map.get(ibottomRight);
		int blackCount = blackBottomRight - blackTopRight - blackBottomLeft + blackTopLeft;
		
		return blackCount;
	}

	public int getHeight(){
		return - topLeft.x + bottomRight.x;
	}
	
	public int getWidth(){
		return bottomRight.y - topLeft.y;
	}

	@Override
	public String toString() {
		return "Rectangle [topLeft=" + topLeft + ", bottomRight=" + bottomRight
				+ "]";
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result
				+ ((bottomRight == null) ? 0 : bottomRight.hashCode());
		result = prime * result + ((topLeft == null) ? 0 : topLeft.hashCode());
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
		Rectangle other = (Rectangle) obj;
		if (bottomRight == null) {
			if (other.bottomRight != null)
				return false;
		} else if (!bottomRight.equals(other.bottomRight))
			return false;
		if (topLeft == null) {
			if (other.topLeft != null)
				return false;
		} else if (!topLeft.equals(other.topLeft))
			return false;
		return true;
	}

	public BlackCount getBlackCountForImg(Image img) {
		return mapBlack.get(img);
	}
	
	
}

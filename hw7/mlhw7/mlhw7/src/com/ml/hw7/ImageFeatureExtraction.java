package com.ml.hw7;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ImageFeatureExtraction {

	List<Rectangle> rectangles;
	List<Rectangle> testRectangles;
	Map<Image, Map<Point, Integer>> cache;
	
	
	public ImageFeatureExtraction() {
		this.rectangles = new ArrayList<Rectangle>();
		this.testRectangles = new ArrayList<Rectangle>();
		this.cache = new HashMap<Image, Map<Point,Integer>>();
	}
	
	public void generateCache(List<Image> list) {
		for(Image img: list){
			Map<Point, Integer> map = new HashMap<Point, Integer>();
			for (int colIdx = 0; colIdx < img.numCols; colIdx++) {
				for (int rowIdx = 0; rowIdx < img.numRows; rowIdx++) {
					Point pij = new Point(rowIdx, colIdx);
					Point pi_1j = new Point(rowIdx - 1, colIdx);
					Point pi_1j_1 = new Point(rowIdx - 1, colIdx - 1);
					Point pij_1 = new Point(rowIdx, colIdx - 1);
					
					int blacki_1j = map.get(pi_1j) == null ? 0 : map.get(pi_1j);
					int blacki_1j_1 = map.get(pi_1j_1) == null ? 0 : map.get(pi_1j_1);
					int blackij_1 = map.get(pij_1) == null ? 0 : map.get(pij_1);
					
					int blackij = blackij_1 + blacki_1j - blacki_1j_1 + img.getImagePixel(rowIdx, colIdx);
					
//				    black(rectangle-diag(ODij)) = black(rectangle-diag(ODi,j-1)) + black(rectangle-diag(ODi-1,j)) 
//                            - black(rectangle-diag(ODi-1,j-1)) + black(pixel Dij)
					
					map.put(pij, blackij);
				}
			}
			cache.put(img, map);
		}
		System.out.println(cache.size());
	}
	
	public void generateRandomRectangles(int size){
		while(rectangles.size() < size){
			Point p1 = generateRandomPoint();
			Point p2 = generateRandomPoint(p1);
			
			Rectangle rect = new Rectangle(p1, p2);
			Rectangle rect2 = new Rectangle(p1, p2);
			rect.calculateBlackCount(cache);
			rectangles.add(rect);
			testRectangles.add(rect2);
		}
	}

	private Point generateRandomPoint(Point p1) {
		int oX = p1.x;
		int oY = p1.y;
		
		int x = (int) getRandomValue(oX + 5, 27);
		int y = (int) getRandomValue(oY + 5, 27);
		
		return (new Point(x, y));
	}

	private Point generateRandomPoint() {
		int x = (int) getRandomValue(0, 27 - 5);
		int y = (int) getRandomValue(0, 27 - 5);
		
		return (new Point(x,y));
	}
	
	private double getRandomValue(double start, double end) {
		Random random = new Random();
		double range = end - start;
		double fraction = range * random.nextDouble();
		return fraction + start;
	}

	public DataSet getHAARFeatureData(List<Image> list) {
		DataSet dataSet = new DataSet(200);
		for(Image img : list){
			Data d = new Data(dataSet.getFeatureSize());
			int feature = 0;
			for(Rectangle r : rectangles){
				BlackCount bc = r.getBlackCountForImg(img);
				if(bc != null){
					d.setFeatureValueAtIndex(feature++, bc.getVecticalCount());
					d.setFeatureValueAtIndex(feature++, bc.getHorizontalCount());
				}
			}
			d.setLabelValue(img.getLabel());
			d.setDataSet(dataSet);
			dataSet.addData(d);
		}
		return dataSet;
	}

	public void calculateRectangleDataForTest(List<Rectangle> rects) {
		for(Rectangle r : rects){
			r.calculateBlackCount(cache);
		}
		this.rectangles = rects;
	}
}

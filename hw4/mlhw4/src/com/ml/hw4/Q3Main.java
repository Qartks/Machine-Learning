package com.ml.hw4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class Q3Main {

	public static void main(String[] args) throws Exception {
		DataSet allData = DataInputer.getData("/Users/kartikeyashukla/Desktop/Masters/Machine Learning/spambase.data");
		
		Collections.shuffle(allData.getData());
//		Collections.shuffle(allData.getData());
		
		
		
		int size = allData.getDataSize();
		int c = (int) (0.05 * size);
		
		DataSet T = new DataSet(allData.getFeatureSize());
		DataSet X = new DataSet(allData.getFeatureSize());
		for(int i = 0; i< size; i++){
			Data d = allData.getData().get(i);
			if( i < c){				
				T.addData(d);
			} else {
				X.addData(d);
			}
		}
		
		activeLearning(size, T, X, 10);
		
	}

	private static void activeLearning(int size, DataSet T, DataSet X, int m) throws Exception {
		
		
		while(T.getDataSize() < 0.5*size){
//			System.out.println("\nNew Round\n");
			

			Collections.shuffle(T.getData());
//			System.out.println(T.getDataSize() + " -> " + X.getDataSize());
			
			AdaBoostImplementation ada = new AdaBoostImplementation(T, m);
			ada.train(T, X, m, true);
			
			double percent = (T.getDataSize()/(double)size) * 100d;
			System.out.println("Percentage of Training Data: " + percent + " : Test Error -> " + ada.aL.error);
			
			HashMap<Integer,Double> map = new HashMap<Integer, Double>();
			for(int i = 0; i<ada.aL.thresVals.size(); i++){
				map.put(ada.aL.indVals.get(i), ada.aL.thresVals.get(i));
			}
			
			List<Double> l = new ArrayList<Double> (map.values());
			Collections.sort(l);
			
			Map<Integer, Double> sort = sortByValue(map);
			
			int counter = 0;
			for(Entry<Integer,Double> e : sort.entrySet()){
				
				T.addData(X.getData().get(e.getKey()));
				X.getData().remove(X.getData().get(e.getKey()));
				
				counter++;
				if(counter > (int)X.getDataSize() * 0.02){
					break;
				}
			}
			
		}
	}
	
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public static Map<Integer, Double> sortByValue(Map unsortMap) {	 
		List list = new LinkedList(unsortMap.entrySet());
	 
		Collections.sort(list, new Comparator() {
			public int compare(Object o1, Object o2) {
				return ((Comparable) ((Map.Entry) (o1)).getValue())
							.compareTo(((Map.Entry) (o2)).getValue());
			}
		});
	 
		Map sortedMap = new LinkedHashMap();
		for (Iterator it = list.iterator(); it.hasNext();) {
			Map.Entry entry = (Map.Entry) it.next();
			sortedMap.put(entry.getKey(), entry.getValue());
		}
		return sortedMap;
	}
}

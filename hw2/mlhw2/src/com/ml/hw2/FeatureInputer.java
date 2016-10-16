package com.ml.hw2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class FeatureInputer {
	
public static List<Feature> getFeaturesList(String fileName) throws Exception{
		
		BufferedReader fReader = null;
		List<Feature> features = null;
		try {
			fReader = new BufferedReader(new FileReader(fileName));
			
			features = new ArrayList<Feature>();
			
			while(true){
				String line = fReader.readLine();
				Feature feature = null;
				
				if(line == null){
					break;		
				}
				
				if(line.trim() == ""){
					break;
				}
				
				String[] values = line.trim().split(":");
				if(values[1].contains("continuous")) {
					feature = new Feature(values[0].trim(), Feature.NUMERICAL);
				} else {
					feature = new Feature(values[0].trim(), Feature.NOMINAL);
				}
				features.add(feature);
			}
		} finally {
			fReader.close();
		}
		return features;
	}

}

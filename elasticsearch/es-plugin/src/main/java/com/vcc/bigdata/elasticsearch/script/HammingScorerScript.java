package com.vcc.bigdata.elasticsearch.script;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.index.fielddata.ScriptDocValues;
import org.elasticsearch.script.AbstractDoubleSearchScript;

import com.vcc.bigdata.elasticsearch.util.FeatureUtil;

import es.fea.ImgFea.ImFeaBin;
import es.fea.ImgFea.ImFeaBinArr;

public class HammingScorerScript extends AbstractDoubleSearchScript {
	private String field;
	private long[] fea;

	public HammingScorerScript(String field, long[] fea) {
		this.field = field;
		this.fea = fea;
	}

	@Override
	public double runAsDouble() {

		BytesRef binaryData = ((ScriptDocValues.BytesRefs) doc().get(field)).get(0);
		ImFeaBinArr feaArr = FeatureUtil.decodeFeaBinary(binaryData);

		if (feaArr == null)
			return -1f;
		double bestScore = 0.0;
		for (int i = 0; i < feaArr.getArrCount(); i++) {
			double score = computeHammintonDistance(fea, feaArr.getArr(i));
			if (score > bestScore)
				bestScore = score;
		}
		return bestScore;
	}

	private double computeHammintonDistance(long[] fea, ImFeaBin arr) {
		double score = 0.0;
		long v1, v2;
		for (int i = 0; i < fea.length; i++) {
			v1 = fea[i];
			v2 = arr.getF(i);

			score += (32 - Long.bitCount(v1 ^ v2));
		}
		return score;
	}

}

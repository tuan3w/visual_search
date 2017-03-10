package com.vcc.bigdata.elasticsearch.script;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.index.fielddata.ScriptDocValues;
import org.elasticsearch.script.AbstractDoubleSearchScript;

import com.vcc.bigdata.elasticsearch.util.FeatureUtil;

import es.fea.ImgFea.ImFea;
import es.fea.ImgFea.ImFeaArr;

public class CosineScorerScript extends AbstractDoubleSearchScript {
	private String field;
	private double[] fea;
	private float queryNorm = 0f;

	public CosineScorerScript(String field, double[] fea) {
		this.field = field;
		this.fea = fea;
		this.queryNorm = computeNorm();
	}

	private float computeNorm() {
		float n = 0f;
		for (int i = 0; i < fea.length; i++)
			n += fea[i] * fea[i];
		if (n == 0)
			return 0.0001f;
		return (float) Math.sqrt(n);
	}

	@Override
	public double runAsDouble() {
		if (queryNorm == 0)
			return 0f;

		BytesRef binaryData = ((ScriptDocValues.BytesRefs) doc().get(field)).get(0);
		ImFeaArr feaArr = FeatureUtil.decodeFea(binaryData);

		if (feaArr == null)
			return -1f;
		double bestScore = 0.0;
		for (int i = 0; i < feaArr.getArrCount(); i++) {
			double score = computeCosineScore(fea, feaArr.getArr(i));
			if (score > bestScore)
				bestScore = score;
		}
		return bestScore;
	}

	private double computeCosineScore(double[] fea, ImFea arr) {
		double score = 0.0;
		double n2 = 0.0;
		double v1, v2;
		for (int i = 0; i < fea.length; i++) {
			v1 = fea[i];
			v2 = arr.getF(i);
			score += v1 * v2;
			n2 += v2 * v2;
		}
		if (n2 == 0)
			return 0.0;
		return score / (queryNorm * Math.sqrt(n2));
	}

}

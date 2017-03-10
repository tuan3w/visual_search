package com.vcc.bigdata.elasticsearch.script;

import java.util.Map;

import org.elasticsearch.script.ExecutableScript;
import org.elasticsearch.script.NativeScriptFactory;
import org.elasticsearch.script.ScriptException;

public class CosineScorerFactory implements NativeScriptFactory {
	public static final String SCRIPT_NAME = "cosine_score";

	public boolean needsScores() {
		return false;
	}

	public String getName() {
		return SCRIPT_NAME;
	}

	public ExecutableScript newScript(Map<String, Object> param) {

		String field = (java.lang.String) param.get("f");
		if (field == null) {
			throw new ScriptException("Field data param [f] is missing", null, null, SCRIPT_NAME, "native");
		}

		double[] fea = (double[]) param.get("fea");
		if (fea == null)
			throw new ScriptException("Param [fea] is missing", null, null, SCRIPT_NAME, "native");

		CosineScorerScript script = new CosineScorerScript(field, fea);
		return script;
	}

}

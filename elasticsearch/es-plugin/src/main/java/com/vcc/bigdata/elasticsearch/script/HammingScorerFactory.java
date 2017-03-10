package com.vcc.bigdata.elasticsearch.script;

import java.util.ArrayList;
import java.util.Map;

import org.elasticsearch.script.ExecutableScript;
import org.elasticsearch.script.NativeScriptFactory;
import org.elasticsearch.script.ScriptException;

public class HammingScorerFactory implements NativeScriptFactory {
	public static final String SCRIPT_NAME = "hamming_score";

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

		@SuppressWarnings("unchecked")
		ArrayList<Number> f = (ArrayList<Number>) param.get("fea");
		if (f == null)
			throw new ScriptException("Param [fea] is missing", null, null, SCRIPT_NAME, "native");
		long[] fea = new long[f.size()];
		for (int i = 0; i < f.size(); i++)
			fea[i] = f.get(i).longValue();

		HammingScorerScript script = new HammingScorerScript(field, fea);
		return script;
	}

}

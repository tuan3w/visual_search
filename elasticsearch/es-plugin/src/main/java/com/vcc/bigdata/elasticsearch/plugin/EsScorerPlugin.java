package com.vcc.bigdata.elasticsearch.plugin;

import java.util.ArrayList;
import java.util.List;

import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.NativeScriptFactory;

import com.vcc.bigdata.elasticsearch.script.CosineScorerFactory;
import com.vcc.bigdata.elasticsearch.script.HammingScorerFactory;

public class EsScorerPlugin extends Plugin implements ScriptPlugin {
	public List<NativeScriptFactory> getNativeScripts() {
		List<NativeScriptFactory> list = new ArrayList<NativeScriptFactory>();
		list.add(new CosineScorerFactory());
		list.add(new HammingScorerFactory());

		return list;
	}

}

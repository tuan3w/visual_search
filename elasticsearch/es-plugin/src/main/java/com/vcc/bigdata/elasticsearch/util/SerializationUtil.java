package com.vcc.bigdata.elasticsearch.util;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Base64;

public class SerializationUtil {

	public static String serialize(Object obj) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ObjectOutputStream os = new ObjectOutputStream(out);
		os.writeObject(obj);
		// return out.toString("ISO-8859-1");
		return Base64.getEncoder().encodeToString(out.toByteArray());

	}

	public static Object deserialize(byte[] bytes) throws IOException, ClassNotFoundException {
		// byte[] bytes = Base64.getDecoder().decode(data);
		ByteArrayInputStream in = new ByteArrayInputStream(bytes);
		ObjectInputStream is = new ObjectInputStream(in);
		return is.readObject();
	}

	public static Object deserialize(String data) throws IOException, ClassNotFoundException {
		byte[] bytes = Base64.getDecoder().decode(data);
		ByteArrayInputStream in = new ByteArrayInputStream(bytes);
		ObjectInputStream is = new ObjectInputStream(in);
		return is.readObject();
	}
}
package com.vcc.bigdata.elasticsearch.util;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.Base64;

import org.apache.lucene.util.BytesRef;
import org.elasticsearch.common.io.stream.StreamInput;

import com.google.protobuf.InvalidProtocolBufferException;

import es.fea.ImgFea.ImFeaArr;
import es.fea.ImgFea.ImFeaBin;
import es.fea.ImgFea.ImFeaBinArr;

public class FeatureUtil {
	public static String encode(ImFeaArr fea) {
		return Base64.getEncoder().encodeToString(fea.toByteArray());
	}

	public static ImFeaArr decodeFea(BytesRef binaryData) {
		StreamInput in = StreamInput.wrap(binaryData.bytes, binaryData.offset, binaryData.length);
		try {
			return ImFeaArr.parseFrom(in);
		} catch (InvalidProtocolBufferException e) {
			// TODO Auto-generated catch block
			System.err.println(e.getMessage());
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println(e.getMessage());

			e.printStackTrace();
		}
		return null;
	}

	public static ImFeaBinArr decodeFeaBinary(BytesRef binaryData) {
		StreamInput in = StreamInput.wrap(binaryData.bytes, binaryData.offset, binaryData.length);
		try {
			return ImFeaBinArr.parseFrom(in);
		} catch (InvalidProtocolBufferException e) {
			// TODO Auto-generated catch block
			System.err.println(e.getMessage());
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.err.println(e.getMessage());

			e.printStackTrace();
		}
		return null;
	}

	public static ImFeaArr decode(byte[] binaryData) {
		// StreamInput in = StreamInput.wrap(binaryData);
		ByteArrayInputStream in = new ByteArrayInputStream(binaryData);
		ObjectInputStream is = null;
		try {
			is = new ObjectInputStream(in);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		// byte[] base64Str = Base64.getDecoder().decode(binaryData);
		try {
			return ImFeaArr.parseDelimitedFrom(is);
		} catch (InvalidProtocolBufferException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			System.err.println(e);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
}

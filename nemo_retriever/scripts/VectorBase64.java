// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

package nrl.benchmark;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;
import org.apache.spark.sql.api.java.UDF1;
import scala.collection.Seq;

/** Match NRL's big-endian float32 base64 representation without Python UDFs. */
public final class VectorBase64 implements UDF1<Seq<Object>, String> {
    @Override
    public String call(Seq<Object> values) {
        if (values == null || values.length() != 768) {
            throw new IllegalArgumentException("Expected a non-null 768-dimensional embedding");
        }
        ByteBuffer bytes = ByteBuffer.allocate(values.length() * 4).order(ByteOrder.BIG_ENDIAN);
        scala.collection.Iterator<Object> iterator = values.iterator();
        while (iterator.hasNext()) {
            float value = ((Number) iterator.next()).floatValue();
            if (!Float.isFinite(value)) throw new IllegalArgumentException("Embedding contains nonfinite values");
            bytes.putFloat(value);
        }
        return Base64.getEncoder().encodeToString(bytes.array());
    }
}

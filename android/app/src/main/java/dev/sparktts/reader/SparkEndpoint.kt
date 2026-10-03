package dev.sparktts.reader

import org.json.JSONObject
import java.net.URI

object SparkEndpoint {
    const val DEFAULT_BASE_URL = "http://SPARK_LAN_IP:8004"

    fun normalizeBaseUrl(raw: String): String {
        var value = raw.trim().trimEnd('/')
        if (value.isEmpty()) throw IllegalArgumentException("Spark-Adresse fehlt")
        if (!value.startsWith("http://") && !value.startsWith("https://")) {
            value = "http://$value"
        }
        val host = URI(value).host
        if (host.isNullOrBlank()) throw IllegalArgumentException("Spark-Adresse ist ungültig")
        return value
    }

    fun speechBody(
        text: String,
        language: String,
        voiceId: String,
        speed: Double,
        stream: Boolean,
        format: String,
    ): ByteArray {
        val json = JSONObject()
        json.put("input", text)
        json.put("language", language)
        json.put("voice", voiceId.ifBlank { "builtin" })
        json.put("speed", speed)
        json.put("stream", stream)
        json.put("response_format", format)
        return json.toString().toByteArray(Charsets.UTF_8)
    }
}

class SpeechCall(
    val endpoint: String,
    val token: String,
    val body: ByteArray,
)

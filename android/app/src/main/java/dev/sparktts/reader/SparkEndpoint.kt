package dev.sparktts.reader

import org.json.JSONObject

object SparkEndpoint {
    const val DEFAULT_BASE_URL = "http://SPARK_LAN_IP:8004"

    fun normalizeBaseUrl(raw: String): String {
        var value = raw.trim().trimEnd('/')
        if (value.isEmpty()) throw IllegalArgumentException("Spark-Adresse fehlt")
        if (!value.startsWith("http://") && !value.startsWith("https://")) {
            value = "http://$value"
        }
        // java.net.URI.host is null for underscores, so the placeholder
        // SPARK_LAN_IP cannot be validated that way.
        val authority = value.substringAfter("://", "").substringBefore("/").substringBefore("?")
        val host = if (authority.startsWith("[")) {
            authority.substringAfter("[").substringBefore("]")
        } else {
            authority.substringBefore(":")
        }
        if (host.isBlank() || !host.any { it.isLetterOrDigit() }) {
            throw IllegalArgumentException("Spark-Adresse ist ungültig")
        }
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

package dev.sparktts.reader

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class Voice(val id: String, val name: String, val builtin: Boolean) {
    companion object {
        val builtin = Voice("builtin", "Eingebaut", true)
    }
}

object VoiceClient {
    fun fetch(baseUrl: String, token: String): List<Voice> {
        val endpoint = SparkEndpoint.normalizeBaseUrl(baseUrl) + "/v1/voices"
        val connection = (URL(endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "GET"
            connectTimeout = 8_000
            readTimeout = 8_000
            instanceFollowRedirects = false
            if (token.isNotBlank()) {
                setRequestProperty("Authorization", "Bearer $token")
            }
        }
        try {
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() }.orEmpty()
            if (code !in 200..299) {
                throw IllegalStateException(httpMessage(code, body))
            }
            val voices = JSONObject(body).getJSONArray("voices")
            val parsed = buildList {
                for (i in 0 until voices.length()) {
                    val item = voices.getJSONObject(i)
                    val id = item.optString("id")
                    if (id.isBlank()) continue
                    add(
                        Voice(
                            id = id,
                            name = item.optString("name").ifBlank { id },
                            builtin = item.optBoolean("builtin", id == "builtin"),
                        ),
                    )
                }
            }
            if (parsed.isEmpty() || parsed.none { it.builtin }) {
                return listOf(Voice.builtin) + parsed
            }
            return parsed
        } finally {
            connection.disconnect()
        }
    }

    private fun httpMessage(code: Int, body: String): String {
        val detail = runCatching { JSONObject(body).optString("detail") }.getOrDefault("")
        return if (detail.isNotBlank()) "HTTP $code: $detail" else "HTTP $code"
    }
}

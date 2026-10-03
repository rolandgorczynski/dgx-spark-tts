package dev.sparktts.reader

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.net.URLEncoder

data class ResolvedText(val text: String, val playable: Boolean)

/**
 * Best-effort readable text when a share contains only a link.
 * X/Twitter status URLs use the public oEmbed endpoint. Other pages use HTML.
 */
object ArticleFetcher {
    fun resolve(raw: String): ResolvedText {
        return when (val classified = ShareText.classify(raw)) {
            is ShareText.Classified.Ready ->
                ResolvedText(classified.text, playable = classified.text.isNotBlank())
            is ShareText.Classified.Fetch -> fetch(classified.url)
        }
    }

    private fun fetch(url: String): ResolvedText {
        val readable = runCatching { readableText(url) }.getOrNull()
        if (!readable.isNullOrBlank() && readable.length >= 20) {
            return ResolvedText(readable, playable = true)
        }
        return ResolvedText(url, playable = false)
    }

    private fun readableText(url: String): String? {
        val host = runCatching { URI(url).host }.getOrNull()?.lowercase()?.removePrefix("www.")
        if (ShareText.isSocialHost(host)) {
            // t.co is only a redirect. x.com/twitter.com can go straight to oEmbed.
            val target = if (host == "t.co") httpGet(url)?.finalUrl ?: url else url
            val tweet = oembed(target)
            if (!tweet.isNullOrBlank()) return tweet
        }
        val page = httpGet(url) ?: return null
        val text = ShareText.htmlToText(page.body)
        return text.ifBlank { null }
    }

    private fun oembed(url: String): String? {
        val endpoint = "https://publish.twitter.com/oembed?omit_script=true&url=" +
            URLEncoder.encode(url, Charsets.UTF_8.name())
        val page = httpGet(endpoint) ?: return null
        val html = runCatching { JSONObject(page.body).optString("html") }.getOrNull()
        if (html.isNullOrBlank()) return null
        val text = ShareText.tweetFromOembedHtml(html)
        return text.ifBlank { null }
    }

    private data class Page(val finalUrl: String, val body: String)

    private fun httpGet(url: String): Page? {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            instanceFollowRedirects = true
            connectTimeout = 8_000
            readTimeout = 8_000
            requestMethod = "GET"
            setRequestProperty("User-Agent", "SparkTTS/1.0 (Android; article fetch)")
            setRequestProperty("Accept", "text/html,application/json;q=0.9,*/*;q=0.8")
        }
        return try {
            val code = connection.responseCode
            if (code !in 200..299) return null
            val body = connection.inputStream.buffered().use { input ->
                val buffer = ByteArray(8 * 1024)
                val out = java.io.ByteArrayOutputStream()
                var total = 0
                while (total < 1_000_000) {
                    val n = input.read(buffer, 0, minOf(buffer.size, 1_000_000 - total))
                    if (n < 0) break
                    out.write(buffer, 0, n)
                    total += n
                }
                out.toByteArray().toString(Charsets.UTF_8)
            }
            Page(connection.url?.toString() ?: url, body)
        } finally {
            connection.disconnect()
        }
    }
}

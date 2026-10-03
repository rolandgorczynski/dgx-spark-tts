package dev.sparktts.reader

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReaderLogicTest {
    @Test
    fun classifyKeepsCaptionAndStripsTrailingLink() {
        val classified = ShareText.classify(
            "Kurzer Post über den Haushalt\nhttps://x.com/someone/status/1",
        )
        assertTrue(classified is ShareText.Classified.Ready)
        assertEquals(
            "Kurzer Post über den Haushalt",
            (classified as ShareText.Classified.Ready).text,
        )
    }

    @Test
    fun classifyFetchesWhenShareIsOnlyAUrl() {
        val classified = ShareText.classify("https://x.com/someone/status/99")
        assertTrue(classified is ShareText.Classified.Fetch)
        assertEquals(
            "https://x.com/someone/status/99",
            (classified as ShareText.Classified.Fetch).url,
        )
    }

    @Test
    fun htmlExtractorPrefersParagraphs() {
        val html = """
            <html><head><title>Ignored title</title>
            <meta property="og:description" content="Short blurb">
            </head><body><article>
            <p>Erster Absatz mit genug Text für eine Vorlesestimme im Test.</p>
            <p>Zweiter Absatz folgt direkt danach und bleibt im Ergebnis.</p>
            </article></body></html>
        """.trimIndent()
        val text = ShareText.htmlToText(html)
        assertTrue(text.contains("Erster Absatz"))
        assertTrue(text.contains("Zweiter Absatz"))
    }

    @Test
    fun oembedHtmlKeepsTheTweetParagraph() {
        val html = """<blockquote><p lang="pl">Dzień dobry &amp; cześć</p>&mdash; User</blockquote>"""
        assertEquals("Dzień dobry & cześć", ShareText.tweetFromOembedHtml(html))
    }

    @Test
    fun socialHostsIncludeXandTwitter() {
        assertTrue(ShareText.isSocialHost("x.com"))
        assertTrue(ShareText.isSocialHost("www.twitter.com"))
        assertTrue(ShareText.isSocialHost("t.co"))
        assertFalse(ShareText.isSocialHost("example.com"))
    }

    @Test
    fun normalizeBaseUrlAddsSchemeAndDropsSlash() {
        assertEquals(
            "http://192.168.1.20:8004",
            SparkEndpoint.normalizeBaseUrl("192.168.1.20:8004/"),
        )
        assertEquals(
            SparkEndpoint.DEFAULT_BASE_URL,
            SparkEndpoint.normalizeBaseUrl("http://SPARK_LAN_IP:8004"),
        )
    }

    @Test
    fun speechBodyEscapesTextAndRequestsOggStream() {
        val raw = SparkEndpoint.speechBody(
            text = "Er sagt \"Hallo\".",
            language = "de",
            voiceId = "anna.wav",
            speed = 1.2,
            stream = true,
            format = "ogg",
        )
        val json = JSONObject(String(raw, Charsets.UTF_8))
        assertEquals("Er sagt \"Hallo\".", json.getString("input"))
        assertEquals("de", json.getString("language"))
        assertEquals("anna.wav", json.getString("voice"))
        assertEquals(1.2, json.getDouble("speed"), 0.001)
        assertTrue(json.getBoolean("stream"))
        assertEquals("ogg", json.getString("response_format"))
    }
}

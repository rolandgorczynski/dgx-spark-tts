package dev.sparktts.reader

/**
 * Turns a share payload into text the TTS API can speak.
 * A lone URL is fetched later. A caption plus a trailing link keeps the caption.
 */
object ShareText {
    private val onlyUrl = Regex("(?i)^https?://\\S+$")
    private val trailingUrl = Regex("(?i)\\s*https?://\\S+\\s*$")

    sealed class Classified {
        data class Ready(val text: String) : Classified()
        data class Fetch(val url: String) : Classified()
    }

    fun classify(raw: String): Classified {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return Classified.Ready("")
        if (onlyUrl.matches(trimmed)) return Classified.Fetch(trimmed)
        val stripped = trimmed.replace(trailingUrl, "").trim()
        if (stripped.isNotEmpty() && stripped != trimmed) return Classified.Ready(stripped)
        return Classified.Ready(trimmed)
    }

    fun htmlToText(html: String): String {
        var source = html
        source = Regex("(?is)<script\\b[^>]*>.*?</script>").replace(source, " ")
        source = Regex("(?is)<style\\b[^>]*>.*?</style>").replace(source, " ")
        val paragraphs = Regex("(?is)<p\\b[^>]*>(.*?)</p>")
            .findAll(source)
            .map { stripTags(it.groupValues[1]) }
            .filter { it.length >= 40 }
            .toList()
        val body = paragraphs.joinToString("\n\n")
        val og = ogDescription(source)
        val title = Regex("(?is)<title[^>]*>(.*?)</title>")
            .find(source)
            ?.let { stripTags(it.groupValues[1]) }
            .orEmpty()
        val chosen = when {
            body.length >= 80 -> body
            og.length >= 40 -> og
            body.isNotBlank() -> body
            else -> title
        }
        return chosen.take(20_000).trim()
    }

    fun tweetFromOembedHtml(html: String): String {
        val paragraph = Regex("(?is)<p\\b[^>]*>(.*?)</p>").find(html)
        val text = if (paragraph != null) stripTags(paragraph.groupValues[1]) else stripTags(html)
        return text.take(20_000).trim()
    }

    fun isSocialHost(host: String?): Boolean {
        if (host.isNullOrBlank()) return false
        val normalized = host.lowercase().removePrefix("www.")
        return normalized == "x.com" ||
            normalized == "twitter.com" ||
            normalized == "t.co" ||
            normalized.endsWith(".x.com") ||
            normalized.endsWith(".twitter.com")
    }

    private fun ogDescription(html: String): String {
        val patterns = listOf(
            "(?is)<meta\\b[^>]*property=[\"']og:description[\"'][^>]*content=[\"'](.*?)[\"']",
            "(?is)<meta\\b[^>]*content=[\"'](.*?)[\"'][^>]*property=[\"']og:description[\"']",
        )
        for (pattern in patterns) {
            val match = Regex(pattern).find(html) ?: continue
            val text = stripTags(match.groupValues[1])
            if (text.isNotBlank()) return text
        }
        return ""
    }

    fun stripTags(html: String): String {
        val noTags = html.replace(Regex("<[^>]+>"), " ")
        return decodeEntities(noTags).replace(Regex("\\s+"), " ").trim()
    }

    fun decodeEntities(input: String): String {
        var text = input
        text = Regex("&#(\\d+);").replace(text) { match ->
            val code = match.groupValues[1].toIntOrNull()
            codePointChar(code) ?: match.value
        }
        text = Regex("&#x([0-9a-fA-F]+);").replace(text) { match ->
            val code = match.groupValues[1].toIntOrNull(16)
            codePointChar(code) ?: match.value
        }
        return text
            .replace("&nbsp;", " ")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&#39;", "'")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
    }

    private fun codePointChar(code: Int?): String? {
        if (code == null || code !in 1..0x10FFFF) return null
        return runCatching { Character.toChars(code).concatToString() }.getOrNull()
    }
}

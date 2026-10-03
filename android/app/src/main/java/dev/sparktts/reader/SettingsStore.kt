package dev.sparktts.reader

import android.content.Context

class SettingsStore(context: Context) {
    private val prefs = context.getSharedPreferences("spark_tts", Context.MODE_PRIVATE)

    var baseUrl: String
        get() = prefs.getString(KEY_URL, SparkEndpoint.DEFAULT_BASE_URL) ?: SparkEndpoint.DEFAULT_BASE_URL
        set(value) = prefs.edit().putString(KEY_URL, value).apply()

    var token: String
        get() = prefs.getString(KEY_TOKEN, "") ?: ""
        set(value) = prefs.edit().putString(KEY_TOKEN, value).apply()

    var language: String
        get() = if (prefs.getString(KEY_LANGUAGE, "de") == "pl") "pl" else "de"
        set(value) = prefs.edit().putString(KEY_LANGUAGE, if (value == "pl") "pl" else "de").apply()

    var voiceId: String
        get() = prefs.getString(KEY_VOICE, "builtin") ?: "builtin"
        set(value) = prefs.edit().putString(KEY_VOICE, value).apply()

    var speed: Float
        get() = prefs.getFloat(KEY_SPEED, 1f).coerceIn(0.5f, 2f)
        set(value) = prefs.edit().putFloat(KEY_SPEED, value.coerceIn(0.5f, 2f)).apply()

    private companion object {
        const val KEY_URL = "base_url"
        const val KEY_TOKEN = "token"
        const val KEY_LANGUAGE = "language"
        const val KEY_VOICE = "voice"
        const val KEY_SPEED = "speed"
    }
}

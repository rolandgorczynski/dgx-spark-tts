package dev.sparktts.reader

import android.content.Context
import android.os.Environment
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** Full WAV download. Playback itself never writes a file. */
object WavArchive {
    fun download(context: Context, baseUrl: String, token: String, body: ByteArray): File {
        val directory = context.getExternalFilesDir(Environment.DIRECTORY_MUSIC) ?: context.filesDir
        directory.mkdirs()
        val file = File(directory, "spark-tts-${System.currentTimeMillis()}.wav")
        val connection = (URL("$baseUrl/v1/audio/speech").openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 0
            doOutput = true
            instanceFollowRedirects = false
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "audio/wav")
            if (token.isNotBlank()) setRequestProperty("Authorization", "Bearer $token")
            setFixedLengthStreamingMode(body.size)
        }
        try {
            connection.outputStream.use { it.write(body) }
            val code = connection.responseCode
            if (code !in 200..299) {
                val err = connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                throw IOException(if (err.isBlank()) "TTS HTTP $code" else "TTS HTTP $code: ${err.take(180)}")
            }
            connection.inputStream.use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
        } catch (error: IOException) {
            file.delete()
            throw error
        } finally {
            connection.disconnect()
        }
        if (file.length() < 44) {
            file.delete()
            throw IOException("Leere Audiodatei")
        }
        return file
    }
}

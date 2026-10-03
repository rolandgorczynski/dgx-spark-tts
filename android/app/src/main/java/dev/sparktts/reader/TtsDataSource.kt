package dev.sparktts.reader

import android.net.Uri
import androidx.media3.common.C
import androidx.media3.datasource.BaseDataSource
import androidx.media3.datasource.DataSource
import androidx.media3.datasource.DataSpec
import java.io.IOException
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.URL

/**
 * POSTs the speech JSON and exposes the response as a forward-only stream.
 *
 * [open] returns [C.LENGTH_UNSET] on purpose. Media3's Ogg reader then uses an
 * unseekable seek map and starts playback on the first pages. A finite length,
 * or a WAV header that pretends the file is gigabytes long, makes the player
 * wait or seek to the end.
 */
class TtsDataSource : BaseDataSource(/* isNetwork = */ true) {
    private var connection: HttpURLConnection? = null
    private var stream: InputStream? = null
    private var openedUri: Uri? = null
    private var opened = false

    override fun open(dataSpec: DataSpec): Long {
        if (dataSpec.position != 0L) {
            throw IOException("TTS-Stream kann nicht spulen")
        }
        val call = TtsRequestStore.pending ?: throw IOException("Keine Vorlese-Anfrage")
        openedUri = dataSpec.uri
        transferInitializing(dataSpec)
        val connection = (URL(call.endpoint).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = 15_000
            readTimeout = 180_000
            doOutput = true
            instanceFollowRedirects = false
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "audio/ogg")
            if (call.token.isNotBlank()) {
                setRequestProperty("Authorization", "Bearer ${call.token}")
            }
            setFixedLengthStreamingMode(call.body.size)
        }
        this.connection = connection
        try {
            connection.outputStream.use { it.write(call.body) }
            val code = connection.responseCode
            if (code !in 200..299) {
                val err = connection.errorStream?.bufferedReader()?.use { it.readText() }.orEmpty()
                throw IOException(errorMessage(code, err))
            }
            stream = connection.inputStream
        } catch (error: IOException) {
            connection.disconnect()
            this.connection = null
            throw error
        }
        opened = true
        transferStarted(dataSpec)
        return C.LENGTH_UNSET.toLong()
    }

    override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
        if (length == 0) return 0
        val input = stream ?: throw IOException("Stream geschlossen")
        val count = input.read(buffer, offset, length)
        if (count == -1) return C.RESULT_END_OF_INPUT
        bytesTransferred(count)
        return count
    }

    override fun getUri(): Uri? = openedUri

    override fun close() {
        try {
            stream?.close()
        } catch (_: IOException) {
            // The player is already tearing the stream down.
        }
        stream = null
        connection?.disconnect()
        connection = null
        if (opened) {
            opened = false
            transferEnded()
        }
    }

    class Factory : DataSource.Factory {
        override fun createDataSource(): DataSource = TtsDataSource()
    }

    private fun errorMessage(code: Int, body: String): String {
        val detail = Regex("\"detail\"\\s*:\\s*\"((?:\\\\.|[^\"])*)\"")
            .find(body)
            ?.groupValues
            ?.get(1)
            ?.replace("\\\"", "\"")
        val text = if (!detail.isNullOrBlank()) detail else body.take(180)
        return if (text.isBlank()) "TTS HTTP $code" else "TTS HTTP $code: $text"
    }
}

package dev.sparktts.reader

import android.Manifest
import android.content.ComponentName
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.SeekBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import androidx.media3.common.MediaItem
import androidx.media3.common.MediaMetadata
import androidx.media3.common.MimeTypes
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.session.MediaController
import androidx.media3.session.SessionToken
import com.google.common.util.concurrent.ListenableFuture
import dev.sparktts.reader.databinding.ActivityMainBinding
import java.io.File
import java.util.Locale
import kotlin.math.roundToInt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var settings: SettingsStore
    private var voices: List<Voice> = listOf(Voice.builtin)
    private var updatingVoices = false
    private var controller: MediaController? = null
    private var controllerFuture: ListenableFuture<MediaController>? = null
    private var pendingPlay = false
    private var activeKey: PlaybackKey? = null

    private val notificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { /* Playback works without a notification; the shade control is optional. */ }

    private val playerListener = object : Player.Listener {
        override fun onIsPlayingChanged(isPlaying: Boolean) {
            binding.playPause.setText(if (isPlaying) R.string.pause else R.string.play)
            if (isPlaying) showStatus("Wiedergabe")
        }

        override fun onPlaybackStateChanged(playbackState: Int) {
            val player = controller ?: return
            if (playbackState == Player.STATE_BUFFERING) {
                val firstSentence = player.currentPosition < 300
                showStatus(if (firstSentence) "Erster Satz wird gesprochen…" else "Nächster Satz…")
            } else if (playbackState == Player.STATE_ENDED) {
                showStatus("Fertig")
                binding.playPause.setText(R.string.play)
            }
        }

        override fun onPlayerError(error: PlaybackException) {
            val detail = error.cause?.message ?: error.message ?: "Wiedergabe fehlgeschlagen"
            showStatus(detail.take(300))
            binding.playPause.setText(R.string.play)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        settings = SettingsStore(this)
        bindSettings()
        bindControls()
        if (Build.VERSION.SDK_INT >= 33) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
        refreshVoices(announceError = false)
        if (savedInstanceState == null) {
            handleIntent(intent)
        } else {
            binding.inputText.setText(savedInstanceState.getString(STATE_DRAFT))
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putString(STATE_DRAFT, binding.inputText.text?.toString().orEmpty())
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent)
    }

    override fun onStart() {
        super.onStart()
        val token = SessionToken(this, ComponentName(this, PlaybackService::class.java))
        val future = MediaController.Builder(this, token).buildAsync()
        controllerFuture = future
        future.addListener({
            if (controllerFuture != future) return@addListener
            val connected = runCatching { future.get() }.getOrNull() ?: return@addListener
            controller = connected
            connected.addListener(playerListener)
            binding.playPause.setText(if (connected.isPlaying) R.string.pause else R.string.play)
            if (pendingPlay) {
                pendingPlay = false
                startPlayback()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onStop() {
        controller?.removeListener(playerListener)
        controller = null
        controllerFuture?.let { MediaController.releaseFuture(it) }
        controllerFuture = null
        super.onStop()
    }

    private fun bindSettings() {
        binding.baseUrl.setText(settings.baseUrl)
        binding.token.setText(settings.token)
        binding.languageToggle.check(if (settings.language == "pl") R.id.langPl else R.id.langDe)
        binding.speed.progress = progressFromSpeed(settings.speed)
        binding.speedLabel.text = speedLabel(settings.speed)
        voices = listOf(Voice(settings.voiceId, settings.voiceId, settings.voiceId == "builtin"))
        applyVoices(voices)
    }

    private fun bindControls() {
        binding.languageToggle.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            settings.language = if (checkedId == R.id.langPl) "pl" else "de"
        }
        binding.speed.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                binding.speedLabel.text = speedLabel(speedFromProgress(progress))
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) = Unit

            override fun onStopTrackingTouch(seekBar: SeekBar?) {
                settings.speed = speedFromProgress(binding.speed.progress)
            }
        })
        binding.voiceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (updatingVoices) return
                settings.voiceId = voices.getOrNull(position)?.id ?: "builtin"
            }

            override fun onNothingSelected(parent: AdapterView<*>?) = Unit
        }
        binding.refreshVoices.setOnClickListener {
            persist()
            refreshVoices(announceError = true)
        }
        binding.baseUrl.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                persist()
                refreshVoices(announceError = true)
            }
        }
        binding.playPause.setOnClickListener { onPlayPause() }
        binding.stop.setOnClickListener {
            controller?.stop()
            controller?.clearMediaItems()
            activeKey = null
            binding.playPause.setText(R.string.play)
            showStatus("Gestoppt")
        }
        binding.save.setOnClickListener { saveWav() }
    }

    private fun handleIntent(intent: Intent?) {
        val raw = sharedText(intent) ?: return
        showStatus("Text wird vorbereitet…")
        lifecycleScope.launch {
            val resolved = withContext(Dispatchers.IO) { ArticleFetcher.resolve(raw) }
            if (isDestroyed) return@launch
            binding.inputText.setText(resolved.text)
            if (!resolved.playable) {
                showStatus("Link erkannt, Text nicht lesbar. Bitte einfügen.")
                return@launch
            }
            if (controller == null) {
                pendingPlay = true
            } else {
                startPlayback()
            }
        }
    }

    private fun onPlayPause() {
        val player = controller
        if (player == null) {
            pendingPlay = true
            showStatus("Player startet…")
            return
        }
        if (player.isPlaying) {
            player.pause()
            return
        }
        val key = currentKey()
        if (player.playbackState == Player.STATE_READY &&
            player.mediaItemCount > 0 &&
            key != null &&
            key == activeKey
        ) {
            player.play()
            return
        }
        startPlayback()
    }

    private fun startPlayback() {
        val player = controller ?: run {
            pendingPlay = true
            return
        }
        val key = currentKey()
        if (key == null) {
            showStatus("Spark-Adresse ist ungültig")
            return
        }
        if (key.text.isEmpty()) {
            showStatus("Kein Text.")
            return
        }
        persist()
        TtsRequestStore.pending = SpeechCall(
            endpoint = key.baseUrl + "/v1/audio/speech",
            token = key.token,
            body = SparkEndpoint.speechBody(
                text = key.text,
                language = key.language,
                voiceId = key.voice,
                speed = key.speed.toDouble(),
                stream = true,
                format = "ogg",
            ),
        )
        activeKey = key
        val title = key.text.lineSequence().firstOrNull()?.trim()?.take(80).orEmpty()
            .ifBlank { getString(R.string.app_name) }
        val item = MediaItem.Builder()
            .setUri(Uri.parse("tts://speech"))
            .setMimeType(MimeTypes.AUDIO_OGG)
            .setMediaMetadata(MediaMetadata.Builder().setTitle(title).build())
            .build()
        player.stop()
        player.clearMediaItems()
        player.setMediaItem(item)
        player.prepare()
        player.play()
        showStatus("Erster Satz wird gesprochen…")
    }

    private fun saveWav() {
        val key = currentKey()
        if (key == null) {
            showStatus("Spark-Adresse ist ungültig")
            return
        }
        if (key.text.isEmpty()) {
            showStatus("Kein Text.")
            return
        }
        persist()
        binding.save.isEnabled = false
        showStatus("Speichert die ganze WAV…")
        val body = SparkEndpoint.speechBody(
            text = key.text,
            language = key.language,
            voiceId = key.voice,
            speed = key.speed.toDouble(),
            stream = false,
            format = "wav",
        )
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { WavArchive.download(this@MainActivity, key.baseUrl, key.token, body) }
            }
            binding.save.isEnabled = true
            result.onSuccess { file ->
                showStatus("Gespeichert: ${file.name}")
                shareFile(file)
            }.onFailure { error ->
                showStatus(error.message ?: "Speichern fehlgeschlagen")
            }
        }
    }

    private fun shareFile(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, getString(R.string.save)))
    }

    private fun refreshVoices(announceError: Boolean) {
        val base = binding.baseUrl.text?.toString().orEmpty()
        val token = binding.token.text?.toString()?.trim().orEmpty()
        lifecycleScope.launch {
            val result = withContext(Dispatchers.IO) {
                runCatching { VoiceClient.fetch(base, token) }
            }
            if (isDestroyed) return@launch
            result.onSuccess { applyVoices(it) }
                .onFailure { error ->
                    if (announceError) showStatus(error.message ?: "Stimmen nicht geladen")
                }
        }
    }

    private fun applyVoices(loaded: List<Voice>) {
        val next = loaded.ifEmpty { listOf(Voice.builtin) }
        updatingVoices = true
        voices = next
        binding.voiceSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            next.map { it.name },
        )
        val index = next.indexOfFirst { it.id == settings.voiceId }.let { if (it < 0) 0 else it }
        binding.voiceSpinner.setSelection(index)
        updatingVoices = false
        settings.voiceId = next[index].id
    }

    private fun currentKey(): PlaybackKey? {
        val text = binding.inputText.text?.toString()?.trim().orEmpty()
        val base = try {
            SparkEndpoint.normalizeBaseUrl(binding.baseUrl.text?.toString().orEmpty())
        } catch (_: IllegalArgumentException) {
            return null
        }
        return PlaybackKey(
            text = text,
            language = if (binding.languageToggle.checkedButtonId == R.id.langPl) "pl" else "de",
            voice = selectedVoiceId(),
            speed = speedFromProgress(binding.speed.progress),
            baseUrl = base,
            token = binding.token.text?.toString()?.trim().orEmpty(),
        )
    }

    private fun selectedVoiceId(): String {
        val index = binding.voiceSpinner.selectedItemPosition
        return voices.getOrNull(index)?.id ?: settings.voiceId
    }

    private fun persist() {
        settings.baseUrl = binding.baseUrl.text?.toString()?.trim().orEmpty()
            .ifBlank { SparkEndpoint.DEFAULT_BASE_URL }
        settings.token = binding.token.text?.toString()?.trim().orEmpty()
        settings.language = if (binding.languageToggle.checkedButtonId == R.id.langPl) "pl" else "de"
        settings.voiceId = selectedVoiceId()
        settings.speed = speedFromProgress(binding.speed.progress)
    }

    private fun showStatus(message: String) {
        binding.status.text = message
    }

    private fun speedFromProgress(progress: Int): Float =
        (0.5f + progress * 0.1f).coerceIn(0.5f, 2f)

    private fun progressFromSpeed(speed: Float): Int =
        ((speed - 0.5f) / 0.1f).roundToInt().coerceIn(0, 15)

    private fun speedLabel(speed: Float): String = String.format(Locale.US, "%.1f×", speed)

    private fun sharedText(intent: Intent?): String? {
        if (intent == null) return null
        val value = when (intent.action) {
            Intent.ACTION_SEND -> intent.getCharSequenceExtra(Intent.EXTRA_TEXT)?.toString()
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            else -> null
        }
        return value?.takeIf { it.isNotBlank() }
    }

    private data class PlaybackKey(
        val text: String,
        val language: String,
        val voice: String,
        val speed: Float,
        val baseUrl: String,
        val token: String,
    )

    private companion object {
        const val STATE_DRAFT = "draft"
    }
}

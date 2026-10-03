package dev.sparktts.reader

/** Process-wide request the player service posts when ExoPlayer opens the stream. */
object TtsRequestStore {
    @Volatile
    var pending: SpeechCall? = null
}

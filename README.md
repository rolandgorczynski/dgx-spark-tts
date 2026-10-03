# Lokales TTS auf dem DGX Spark

Chatterbox Multilingual **V3** (Resemble AI, 0,5B, **MIT**) als HTTP-API.
Deutsch und Polnisch stehen beide in der offiziellen 23-Sprachen-Liste
(`de`, `pl`). Kein Cloud-TTS, kein Ollama.

Die Android-App unter `android/` nimmt Text per Teilen (X, Browser, Artikel)
und spielt **Ogg Opus** ab, sobald der erste Satz fertig ist. Die Stimme
kommt vom Spark, nicht von einer Datei auf dem Telefon.

Der Container wird **auf dem Spark gebaut**. Es gibt kein fertiges Hub-Image.
Basis `nvidia/cuda:13.0.3-runtime-ubuntu22.04` (linux/arm64, SBSA, passend zu Grace)
und PyTorch `2.10.0+cu130` / torchaudio `2.10.0+cu130` (aarch64, cp310) wurden
am 2026-10-03 geprüft. Der lokale Image-Name `spark-tts:chatterbox-mtl-v3`
entsteht erst durch den Build.

Dieser Stack wurde hier **nicht** auf einem DGX Spark ausgeführt.

## Was du brauchst

- Ubuntu auf dem Spark, Docker, NVIDIA-Treiber mit CUDA 13 (Community-Rezept: Treiber 580+).
- NVIDIA Container Toolkit (`nvidia-ctk runtime configure --runtime=docker`, danach Docker neu starten).
- Internet für den einmaligen Image-Build und den Modell-Download.
- Optional: `cp .env.example .env` (kein Secret nötig; `HF_TOKEN` leer lassen).
- Telefon und Spark im selben LAN, Port **8004** vom Telefon erreichbar.

## Start

```bash
cd spark-tts
docker compose up -d --build
docker compose logs -f tts
```

Erster **Build**: oft 20–40 Minuten (APT, Git, PyTorch-Wheel).
Erster **Start** lädt danach nur die nötigen Dateien von Hugging Face
`ResembleAI/chatterbox`: `t3_mtl23ls_v3.safetensors` (~2,1 GB), `s3gen.pt`
(~1,1 GB), `ve.pt`, Tokenizer, `conds.pt`. Zusammen etwa **3,2 GB**,
typisch einige Minuten plus 1–2 Minuten Laden in den Speicher.
Der Healthcheck wartet bis zu 15 Minuten (`start_period`). Danach:

```bash
curl -sf http://127.0.0.1:8004/health
```

`status` muss `ok` sein und `cuda` `true`. Ohne sichtbare GPU beendet sich der Prozess (kein stiller CPU-Fallback). `TTS_ALLOW_CPU=1` nur für einen langsamen Probelauf. Cache liegt im Volume `hf_cache`,
ein zweiter Start lädt nicht neu.

Auf dem GB10 (aarch64) kann `resemble-perth` die Klasse `PerthImplicitWatermarker` auf `None` lassen, weil die native Erweiterung nicht lädt. Chatterbox ruft sie trotzdem auf und der Start starb bisher mit `TypeError: 'NoneType' object is not callable` in `load_model`. Dieser Stand setzt vorher einen Durchreich-Watermarker: der Container startet, die Synthese liefert Audio, nur ohne Perth-Wasserzeichen. In den Logs steht dann eine Warnung, kein Abbruch.

Die Spark-IP für das Telefon:

```bash
ip -4 addr show | awk '/inet / && $2 !~ /^127\./ {print $2}'
```

In der App die Adresse eintragen, zum Beispiel `http://192.168.1.20:8004`.
Wenn `ufw` aktiv ist: `sudo ufw allow 8004/tcp`.

## Test (auf dem Spark)

Deutsch, ganze WAV-Datei:

```bash
curl -sS -X POST http://127.0.0.1:8004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Guten Tag. Das ist ein Test der lokalen Sprachausgabe auf dem Spark.","language":"de"}' \
  --output de.wav
```

Polnisch:

```bash
curl -sS -X POST http://127.0.0.1:8004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Dzień dobry. To jest test lokalnej syntezy mowy na Spark.","language":"pl"}' \
  --output pl.wav
```

Dieselbe Route heißt auch `POST /tts` und akzeptiert `text` statt `input`.

## Stream (sofort hörbar)

Langer Text wird an Satzgrenzen und Zeilenumbrüchen geschnitten. Der **erste
Satz** geht allein ins Modell, der Rest in Stücken bis `TTS_CHUNK_CHARS`
(Standard 400). Das Modell selbst erzeugt keinen Token-Stream.

Für Abspieler: `stream: true` und `response_format: "ogg"`. Das ist **Ogg Opus**
über chunked HTTP, ohne `Content-Length`. ExoPlayer behandelt den Strom als
nicht spulbar und beginnt mit den ersten Ogg-Seiten, während die nächsten Sätze
noch laufen. `ffmpeg` (`libopus`, `atempo`) sitzt schon im Image.

```bash
curl -N -sS -X POST http://127.0.0.1:8004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Erster Satz. Zweiter Satz kommt danach.","language":"de","stream":true,"response_format":"ogg"}' \
  --output de-stream.ogg
```

`curl -N` schreibt, sobald der erste Satz kodiert ist. Ein einzelner kurzer Satz
hat davon keinen Vorteil.

`stream: true` mit `response_format: "wav"` bleibt für `curl` erhalten. Der
WAV-Header trägt eine Platzhalterlänge (`0x7FFFFF00`), weil die echte Länge
erst am Ende feststeht. Media3 liest diese Länge als Dateiende (mehrere
Stunden) und sucht nicht sauber bis zum Socket-Ende. Die App nutzt deshalb Ogg,
nicht WAV. MP3 gibt es nur ohne Stream.

Antwort-Header beim Stream: `Cache-Control: no-store`, `Accept-Ranges: none`,
`X-Accel-Buffering: no` (kein Puffer, falls später ein Reverse-Proxy davor steht).

## Sprache, Stimme, Tempo

| Feld | Bedeutung |
| --- | --- |
| `language` | `de` oder `pl` (auch `en` und 20 weitere). Fehlt das Feld, gilt `TTS_LANGUAGE` (Standard `de`). |
| `voice` | `builtin` oder der Dateiname aus `GET /v1/voices`. |
| `speed` | `0.5`–`2.0`, Standard `1.0`. |

`GET /v1/languages` listet die Sprachen. `GET /v1/voices` liefert die eingebaute
Stimme plus Dateien in `./voices` (wav, mp3, flac, ogg):

```json
{"voices":[{"id":"builtin","name":"Eingebaut","builtin":true},{"id":"anna.wav","name":"anna","builtin":false}]}
```

Die eingebaute Stimme ist der Default des Modells (`conds.pt`), kein eigener
deutscher oder polnischer Sprecher. Für einen muttersprachlichen Klang eine
kurze, saubere Referenz (wenige Sekunden) nach `./voices/` legen. Die App lädt
die Liste selbst. Nichts aufs Telefon kopieren. Nur der Dateiname, kein Pfad.

`generate()` hat **kein** Sprechtempo. `cfg_weight` verschiebt die Prosodie und
ist kein Tempo-Regler; der Default `0.5` bleibt. `speed` streckt die PCM-Daten
mit ffmpeg `atempo` (Tonhöhe bleibt). `1.0` lässt die Samples unverändert.
Weitere Felder: `exaggeration` (0.5), `cfg_weight` (0.5), `temperature` (0.8), `seed`.

OpenAPI: `http://127.0.0.1:8004/docs`.

## Optionaler LAN-Token

`TTS_API_TOKEN` in `.env`. Leer lassen, dann ist die API offen.

```bash
curl -sS http://127.0.0.1:8004/v1/voices -H 'Authorization: Bearer geheim'
```

Dasselbe gilt für `X-API-Key`. `GET /health` bleibt ohne Token, damit der
Docker-Healthcheck nicht scheitert. In der App das Feld „Token“ füllen.

## Android

Ordner `android/` in Android Studio öffnen (JDK 17, Android SDK 35) und auf
dem Gerät starten. Oder Debug-APK ohne Play Store:

```bash
cd android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Ablauf:

1. App öffnen, Spark-Adresse auf `http://<spark-ip>:8004` setzen (Vorgabe im Feld: `http://SPARK_LAN_IP:8004`). Optional den Token eintragen.
2. „Stimmen laden“. Deutsch oder Polski wählen, Stimme und Tempo einstellen.
3. In X (oder einer anderen App) **Teilen → Spark Vorlesen**. Enthält die Freigabe schon den Text, startet die Wiedergabe. Kommt nur ein Link, holt die App den Text (bei X über oEmbed, sonst aus dem HTML). Klappt das nicht, bleibt das Feld zum Einfügen offen.
4. Abspielen / Pause / Stopp. Die Wiedergabe läuft im Player-Dienst weiter, wenn du zu X zurückgehst. Die Benachrichtigung braucht die Mitteilungs-Erlaubnis.
5. „Speichern“ ist nur der Nebenweg: eine ganze WAV, danach die Teilen-Ansicht. Der normale Weg schreibt keine Datei.

Klartext-HTTP ist erlaubt, weil der Spark im LAN kein TLS spricht. Tempo gilt
beim nächsten Start, nicht mitten in einem laufenden Stück.

## Speicher

Das Modell ist 0,5B. Rechne mit wenigen GB GPU-Speicher, nicht mit dem ganzen
128-GB-Unified-Memory. Der Spark teilt sich RAM und GPU-Speicher; 128 GB sind
mehr als genug. FP32, kein BF16, Qualität vor Tempo.

## Grenzen

- Polnisch ist erstklassig in dem Sinn, dass es dieselbe Modellliste wie Deutsch ist.
  Es gibt kein eigenes Single-Language-Pack für `pl` oder `de` (die gibt es nur für
  Chinesisch, Spanisch, Portugiesisch, Hindi). Die Aussprache kommt aus dem
  allgemeinen Mehrsprachmodell.
- Der offizielle Loader paart T3-V3 mit `s3gen.pt`, nicht mit der separat
  liegenden Datei `s3gen_v3.pt`. Dieser Stack folgt dem offiziellen Loader.
- Läuft Perths impliziter Watermarker, bleiben die Ausgaben mit einem kaum hörbaren Wasserzeichen versehen (kein Lizenzverbot). Auf dem Spark fällt der Import oft aus; dann ist das Audio ohne Wasserzeichen, siehe den Hinweis oben beim Start.
- Ein Aufruf im Modell endet bei 1000 Speech-Tokens (grob 40 Sekunden).
  Längerer Text wird bei ~400 Zeichen an Satzgrenzen geschnitten (`TTS_CHUNK_CHARS`).
  Der erste Satz eines Streams ist davon ausgenommen, damit früher Audio kommt.
- An den Schnitten bleibt ohne Crossfade ein kleiner Knackser möglich. Der Server
  mischt 20 ms über. `speed` ungleich 1.0 kommt obendrauf und kann die Übergänge
  leicht verändern.
- `zh`/`ja`-Hilfsbibliotheken (pkuseg, pykakasi) sind nicht installiert.
  Deutsch und Polnisch brauchen sie nicht.
- PyTorch 2.10+cu130 warnt auf dem GB10: `Found GPU0 NVIDIA GB10 which is of cuda capability 12.1` und nennt als Maximum 12.0. Das ist erwartet. Das Wheel bringt sm_120-Kernel mit, die auf sm_121 laufen. `torch.cuda.get_arch_list()` muss dafür kein `sm_121` enthalten, und die Warnung ist kein Grund, Torch neu zu bauen. Bricht die Inferenz mit „no kernel image“ ab, liegt das falsche Wheel im Image (CPU-Build oder ein Torch ohne sm_120).

```bash
docker compose exec tts python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
```

## Tests ohne GPU

```bash
pip install -r requirements-dev.txt
PYTHONPATH=. pytest
```

Die Tests erzeugen PCM lokal und prüfen, dass Ogg-Seiten ankommen, bevor der
zweite Satz an den Encoder geht.

## Ollama

Nicht im Stack. Chatterbox ist kein Ollama-Modell. Qwen3-TTS läuft auf dem
Spark (eigenes Image, z. B. Community `martinb78/faster-qwen3-tts-dgx-spark`),
kann aber offiziell kein Polnisch. Fish Audio S2 Pro ist in öffentlichen
Vergleichen oft stärker, listet Deutsch aber als Tier 2 und Polnisch nur in
der langen Restliste, und die Gewichte sind Research-Lizenz (nicht kommerziell).

## Lizenz

Modell und Code von Resemble: MIT. Dieser kleine Server ebenfalls nutzbar ohne
Extra-Lizenz. Keine API-Keys für das Modell.

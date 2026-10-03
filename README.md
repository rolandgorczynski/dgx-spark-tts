# Lokales TTS auf dem DGX Spark

Chatterbox Multilingual **V3** (Resemble AI, 0,5B, **MIT**) als HTTP-API.
Deutsch und Polnisch sind beide in der offiziellen 23-Sprachen-Liste
(`de`, `pl`), nicht als Nachgedanke. Kein Cloud-TTS, kein Ollama, keine Android-App.

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

Langer Text als WAV-Stream (satzweise, nicht tokenweise). `curl -N` schreibt,
sobald der erste Satz fertig ist. Der WAV-Header trägt eine Platzhalterlänge
(`0x7FFFFF00`); zum Abspielen die Datei nach Ende des Requests öffnen.
Kurze Sätze haben davon keinen Vorteil.

```bash
curl -N -sS -X POST http://127.0.0.1:8004/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Erster Satz. Zweiter Satz kommt danach.","language":"de","stream":true}' \
  --output de-stream.wav
```

MP3 nur ohne Stream: `"response_format":"mp3"`.
Sprachen: `GET /v1/languages`. OpenAPI: `http://127.0.0.1:8004/docs`.

## Sprache und Stimme

| Feld | Bedeutung |
| --- | --- |
| `language` | `de` oder `pl` (auch `en` und 20 weitere). Fehlt das Feld, gilt `TTS_LANGUAGE` (Standard `de`). |
| `voice` | weglassen oder `builtin`: eingebaute Stimme aus `conds.pt`. |

Die eingebaute Stimme ist der Default des Modells, kein deutscher oder polnischer Sprecher.
Für einen muttersprachlichen Klang eine kurze, saubere Referenz (wenige Sekunden, WAV/MP3/FLAC)
nach `./voices/` legen und `"voice":"dateiname.wav"` schicken. Nur der Dateiname, kein Pfad.

Weitere Felder: `exaggeration` (0.5), `cfg_weight` (0.5), `temperature` (0.8), `seed`.

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
- Ausgaben sind mit Perth-Watermark versehen (kaum hörbar, kein Lizenzverbot).
- Ein Aufruf im Modell endet bei 1000 Speech-Tokens (grob 40 Sekunden).
  Längerer Text wird bei ~400 Zeichen an Satzgrenzen geschnitten (`TTS_CHUNK_CHARS`).
- Stream ist satzweise WAV. An den Schnitten kann ein kleiner Knackser bleiben.
  Ohne `stream` werden die Stücke mit 20 ms Crossfade zusammengehängt.
- `zh`/`ja`-Hilfsbibliotheken (pkuseg, pykakasi) sind nicht installiert.
  Deutsch und Polnisch brauchen sie nicht.
- Wenn `torch.cuda.get_arch_list()` kein `sm_121` enthält, bricht die Inferenz
  mit „no kernel image“ ab. Dann ist das falsche Torch-Wheel im Image.

```bash
docker compose exec tts python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
```

## Ollama

Nicht im Stack. Chatterbox ist kein Ollama-Modell. Qwen3-TTS läuft auf dem
Spark (eigenes Image, z. B. Community `martinb78/faster-qwen3-tts-dgx-spark`),
kann aber offiziell kein Polnisch. Fish Audio S2 Pro ist in öffentlichen
Vergleichen oft stärker, listet Deutsch aber als Tier 2 und Polnisch nur in
der langen Restliste, und die Gewichte sind Research-Lizenz (nicht kommerziell).

## Lizenz

Modell und Code von Resemble: MIT. Dieser kleine Server ebenfalls nutzbar ohne
Extra-Lizenz. Keine API-Keys.

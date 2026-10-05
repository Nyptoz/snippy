# Snippy

A Discord voice-clipping bot that lives in your call.

Snippy joins a voice channel, listens, and cuts a clip when you ask it to —
by saying *"hey Snippy, clip this"* out loud, or with a slash command. It
keeps a rolling recording so you can go back and grab something from twenty
minutes ago, and it can separate speakers when you only want one of them.

Clips can be up to two minutes long.

---

## Part 1 — Set up the bot on Discord's side

This is all done once, in the [Discord Developer Portal](https://discord.com/developers/applications).
No coding, and nothing here needs to be repeated.

### 1. Create the application

Go to <https://discord.com/developers/applications> and click **New Application**.
Name it `Snippy`, give it a short description, and click **Create Application**.

### 2. Create the bot and get its token

In the left sidebar click **Bot**, then **Reset Token**, then **Copy**.

The token is shown exactly once. Put it in a file called `.env` next to the
code:

```
DISCORD_TOKEN=paste-your-token-here
DISCORD_GUILD_ID=your-server-id
```

You get the server ID by right-clicking the server, choosing **Copy Server ID**
and turning on Developer Mode in Discord's settings.

### 3. Turn on the privileged intents

Still on the **Bot** page, scroll to **Privileged Gateway Intents**:

| Intent | Setting | Why |
|---|---|---|
| **Message Content Intent** | **ON** | Needed for the typed trigger phrases. Without it the bot connects but typed triggers never fire. |
| **Server Members Intent** | **ON** | Needed so `/clip person:@Ken` autocomplete works and so clip embeds show names instead of raw user IDs. |
| **Presence Intent** | OFF | Snippy does not use it. |

Both of these are free for a bot in a handful of servers. Discord requires
verification before it will grant them to an app with a large audience.

### 4. Create a channel for clips

In your server, make a text channel called `snippy-clips`. Snippy looks it up
by name, so this spares you from granting it **Manage Channels** just so it can
create one itself. Set a different name with
`/snippy config delivery clip_channel_name:<name>`.

### 5. Invite the bot

Go to **OAuth2 → URL Generator** on your application's page.

- **Scopes:** `bot` and `applications.commands`
- **Permissions:** View Channel, Send Messages, Send Messages in Threads, Create
  Public Threads, Embed Links, Attach Files, Read Message History, Connect,
  Speak, Use Slash Commands

That produces an invite link like this one:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=311388392448&scope=bot%20applications.commands
```

Open it, choose your server, and authorise. Snippy appears in the member list
but stays offline until you actually start it (Part 2).

`311388392448` is the permission set above, and it deliberately **excludes
Manage Channels**. If you would rather Snippy create `#snippy-clips` itself,
add Manage Channels and use `permissions=311388392464` instead.

---

## Part 2 — Run it

You need **Python 3.11 or 3.12** and **ffmpeg**.

```bash
# Debian/Ubuntu
sudo apt install ffmpeg python3.12 python3.12-venv
# macOS
brew install ffmpeg
```

Then, from a clone of this repository:

```bash
./install.sh              # installs, asks about spoken triggers, starts it
```

The installer creates a virtualenv, installs the dependencies, checks ffmpeg,
and registers a **systemd user service** so Snippy keeps running after you log
out of SSH. It also runs `loginctl enable-linger` for you — that step is what
makes the service survive your session ending.

On a host without systemd:

```bash
./install.sh --no-systemd
./scripts/run-daemon.sh   # setsid + nohup + pidfile
```

Useful day-to-day commands:

```bash
python -m snippy --check      # validate the install without connecting
journalctl --user -u snippy -f # follow the logs
systemctl --user restart snippy
```

### Spoken triggers

Spoken triggers need speech recognition, which is an optional extra:

```bash
./install.sh --asr
```

It installs `faster-whisper` and a ~75 MB `tiny.en` model, and runs CPU-only.
Until you do, slash commands and typed triggers still work. An admin can turn
speech recognition on per server with `/snippy config asr enabled:True`.

---

## How it works

Discord sends Opus packets over the gateway. `discord-ext-voice-recv` hands
them to a sink which decodes them and feeds three places at once:

| Buffer | Holds | Used by |
|---|---|---|
| **Mix ring** | 60 s of everyone mixed, in RAM | `/clip` — instant |
| **Stems** | 20 s per speaker, up to 8 speakers | `solo`, `duet`, `stems`, `duck` |
| **Archive** | Everything, as 32 kbps Opus on disk | `/reclip`, `/replay` |

The archive is what makes "clip what I said an hour ago" possible — a RAM ring
cannot answer that. It is pruned by retention time and a disk quota, and it
self-erases.

When you ask for a clip, ffmpeg renders the requested window with a filter
chain chosen by style, plus silence trimming, loudness normalisation and edge
fades. The rendered file is uploaded and then deleted; only the archive copy
persists.

---

## Commands

### Clipping

| Command | What it does |
|---|---|
| `/clip` | Clip the last few seconds. `window:15s`, `style:One speaker person:@Ken` |
| `/reclip when:2 hours ago duration:30s` | Cut from the archive. Accepts `5m`, `2 hours ago`, `yesterday 8pm`, `2:30 into the call` |
| `/replay when:10m` | Play an earlier moment back into the voice channel |
| `/when` | A picture of who was talking, per speaker lane |
| `/search query:"that one bit"` | Search transcripts of past clips |

**Styles:** `Everyone (mix)`, `Per speaker (stems)`, `One speaker (solo)`,
`Two speakers (duet)`, `Speech only (active)`, `Speaker in front (duck)`.

Discord has no "list of people" option, so a duet takes two named members:

```
/clip style:Two speakers person:@Ken person2:@Alex
```

Omitting `person` on a solo means *you*. A duet given only one speaker is
rejected rather than guessed at.

### Session

| Command | What it does |
|---|---|
| `/session join` | Ask Snippy to start listening in your channel |
| `/session leave` | Ask Snippy to stop |
| `/session status` | What it is doing right now |

### Admin

All admin commands require **Manage Server**, and answers are ephemeral.

| Command | What it does |
|---|---|
| `/snippy settings` | Show the current configuration |
| `/snippy config audio` | Clip length, default style, trim, normalise |
| `/snippy config delivery` | Where clips are posted, thread per session |
| `/snippy config join` | Join policy and channels |
| `/snippy config triggers` | Spoken phrases, typed triggers, fuzzy threshold |
| `/snippy config archive` | Retention, quota, per-speaker archiving |
| `/snippy config privacy` | Solo/stems allowed, consent banner |
| `/snippy config asr` | Speech recognition and transcripts |
| `/snippy config safety` | Rate limits, retention ceiling |
| `/snippy ignore member:@Ken recording:True` | Drop someone from all audio |
| `/snippy storage trim:True` | Show disk use, delete past the window |
| `/snippy reset confirm:True` | Back to defaults |

---

## Two things that surprise people

**`archive_stems` is off by default.** Per-speaker rings only hold 20 seconds,
so `solo`, `duet` and `stems` against anything older than that have no
per-speaker audio to work from. Turn it on if you want them:

```
/snippy config archive archive_stems:True
```

That costs one extra ffmpeg encoder per active speaker.

**Spoken triggers need the ASR extra.** See Part 2. `/session status` tells you
whether recognition is available.

---

## Configuration

`config/default.toml` is the baseline for every server, and documents the valid
range of every knob. Anything an admin changes is stored per server in SQLite
and overrides the file.

Environment variables, all optional except the first:

| Variable | Meaning |
|---|---|
| `DISCORD_TOKEN` | **Required.** Bot token. |
| `DISCORD_GUILD_ID` | Register slash commands in this one server. Recommended: commands then appear in about a second instead of up to an hour. |
| `SNIPPY_DATA_DIR` | Where the database and archive live. Default `./data`. |
| `SNIPPY_CONFIG` | Path to the config file. |
| `SNIPPY_LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

Command-line flags override the environment: `--token`, `--guild-id`,
`--data-dir`, `--config`, `--log-level`, `--asr`, `--check`.

---

## Development

```bash
./.venv/bin/python -m pytest -q
```

The suite is fully offline — no Discord connection, no network. Audio is
synthesised and ffmpeg is invoked directly. `tests/test_cogs.py` walks the
entire slash-command tree and serialises it, which is what catches discord.py
API drift before the bot meets a gateway.

## Limitations

- The bot has been verified against ffmpeg and at the audio level, but has not
  yet been run against a live Discord gateway. Voice connect and upload are
  unexercised.
- `scripts/install.sh`, `scripts/run-daemon.sh` and `systemd/snippy.service`
  target Linux. They have not been executed on a Windows host.
- `Voice State` is a non-privileged intent that is on by default; the two
  privileged intents that matter are the ones in Part 1, step 3.
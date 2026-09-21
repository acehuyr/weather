# Hosting this project

Streamlit Community Cloud is the right target: free, no card, and it runs
Streamlit apps natively. The whole stack was chosen to fit inside it.

---

## Before you start — why this will fit

Community Cloud gives roughly **1 GB of RAM and no GPU**. That is the
constraint that shaped several earlier decisions:

| Choice | Why it matters for hosting |
|---|---|
| Groq for the language model | The model runs on Groq's hardware. Your app ships a ~56-character key, not 5 GB of weights. |
| `sentence-transformers` left commented in `requirements.txt` | Installing it pulls PyTorch, which alone exceeds the memory budget. The default hashed TF-IDF retriever is pure numpy. |
| Open-Meteo and OpenStreetMap | No keys, no quotas to manage in production. |

**Do not uncomment `sentence-transformers` for a hosted deployment.** It works
locally and will fail to build or get OOM-killed on the free tier.

---

## 1. Put the project on GitHub

It is not a git repository yet.

```bash
git init
git add .
git commit -m "Weather Insight Engine"
```

Check what you are about to publish **before** pushing:

```bash
git status --short
```

`.env` must not appear. It is gitignored, and it holds your Groq key. If you
ever see it listed, stop and fix `.gitignore` first — a key pushed to GitHub
is a key you have to revoke.

Then create an empty repository on GitHub and push:

```bash
git remote add origin https://github.com/<you>/<repo>.git
git branch -M main
git push -u origin main
```

A **public** repository is fine and is what the free tier expects. Nothing
secret is in the code; the key lives in step 3.

---

## 2. Create the app

1. Go to **share.streamlit.io** and sign in with GitHub
2. **New app** → pick your repository
3. Branch `main`, main file path:

```
src/ui/app.py
```

4. Click **Deploy**

The first build takes a few minutes while it installs
`requirements.txt`.

---

## 3. Add your key as a secret

`.env` is gitignored, so the deployed app has no key until you supply one.

In the app's **Settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
LLM_BACKEND = "groq"
```

`config/settings.py` reads environment variables first and falls back to
`st.secrets`, so this works without any code change. Save, and the app
restarts automatically.

Without this the app still runs — offline mode, rule-based planner, template
answers, with all analysis, anomaly detection, retrieval and the map intact.
It simply will not converse.

---

## 4. What happens on first load

The RAG index is gitignored (`data/vectorstore/`), so it does not exist on a
fresh deployment. `Retriever` detects this and builds it on first use —
about a second for 40 chunks. No manual step, but the very first question
after a cold start is slightly slower.

---

## Known limits of the free tier

**It sleeps.** An app with no traffic is suspended and takes ~30 seconds to
wake. Before a demo or viva, **open it five minutes early** so it is warm.
This is the single most common way a live demo goes wrong.

**Groq's free tier meters tokens per minute** (8,000 at the time of writing).
That is roughly three questions a minute. Fine for a demo, and shared across
everyone using your deployed link — so if you circulate the URL widely,
expect throttling.

**Everything is public.** The repository is public and anyone with the link
can use the app, spending your Groq quota. That is acceptable for a college
project; do not put anything private in the knowledge base.

---

## Verify the deployment

Once it is live, check these in order:

1. The sidebar **model** row shows `openai/gpt-oss-120b`, not `not connected`
   — if it says not connected, the secret is missing or misspelled
2. **Dashboard** loads current conditions and the departure scale
3. **Map** renders tiles with nearby markers
4. **Ask** answers a question, and a *second* question works without clearing

If the model row is wrong, check Settings → Secrets for a typo. A `404 model
does not exist` is a retired model ID, not a bad key — run
`python scripts/list_models.py` locally and update `GROQ_MODEL`.

---

## If you would rather not use GitHub

Streamlit Cloud requires a repository. For a purely local demo:

```bash
streamlit run src/ui/app.py
```

To show it on another machine on the same network:

```bash
streamlit run src/ui/app.py --server.address 0.0.0.0
```

Then open `http://<your-ip>:8501` from that machine. No hosting, no public
URL, and it stops when you close the terminal — but nothing can rate-limit
or suspend it mid-demo.

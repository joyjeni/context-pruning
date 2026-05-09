# Kaggle Secrets Setup

Use Kaggle Secrets for API keys. Do not paste keys into public notebook cells,
Markdown, committed config files, or GitHub.

## 1. Add the secret in Kaggle

In your Kaggle notebook:

1. Open the right sidebar.
2. Click **Add-ons**.
3. Click **Secrets**.
4. Add a new secret:

```text
Label: GEMINI_API_KEY
Value: your new Google AI Studio / Gemini API key
```

5. Enable notebook access for the secret.

## 2. Verify the secret in a notebook cell

Write this code in a Kaggle notebook code cell:

```python
from kaggle_secrets import UserSecretsClient

secret_label = "GEMINI_API_KEY"
secret_value = UserSecretsClient().get_secret(secret_label)

print("API key loaded:", bool(secret_value))
```

Expected output:

```text
API key loaded: True
```

Never run `print(secret_value)`.

## 3. How this repo uses the secret

The Kaggle runner notebook reads `GEMINI_API_KEY` and writes a private runtime
config file:

```text
/kaggle/working/configs/secrets.toml
```

with:

```toml
[gemma]
api_key = "..."
```

That file is generated inside the Kaggle runtime and should not be committed.

## 4. If a key was exposed

If an API key was shown in a screenshot, notebook output, GitHub file, or chat,
delete/revoke it immediately in Google AI Studio and create a new key:

```text
https://aistudio.google.com/apikey
```

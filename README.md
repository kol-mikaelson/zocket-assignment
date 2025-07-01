
# Zocket Technical Assignment

This a repositary containg submission for assesment to Zocket, the project is a Ad rewriting agent using Llama 4 at the backend to rewrite the Ad, the full technical report can be found here: 
https://docs.google.com/document/d/1Qm-TLQTtfm8JcfDsrKeANCHt9-5GrD6oZJTZ02my94E/edit?usp=sharing
## Demo

https://youtu.be/8ptfZKNq71A


## Run Locally
### Note to run the project Locally you need a Open Router/OPENAI API Key so that the LLM Functionality works
Clone the project

```bash
  git clone https://github.com/kol-mikaelson/zocket-assignment.git

Go to the project directory

```bash
  cd zocket-assignment
```
setup a .env file with api key
```bash
  OPENROUTER_API_KEY=KEY
```
Install dependencies

```bash
  pip install -r requirements.txt
```

Run the backend

```bash
  python main.py
```

Run the frontend

```bash
  python flask_frontend/app.py
```
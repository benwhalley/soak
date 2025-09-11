
# llemma: tools for psychological research, education and practice

This project is a WIP but currently packages one tool

- soak: a DAG based method to describe text-analysis pipelines for qualitative/thematic analysis.



# SETUP

On OS X or linux:

Install UV: https://docs.astral.sh/uv/getting-started/installation


Clone the repo:

```
git clone https://github.com/benwhalley/llemma
cd llemma
```


Install the package:

```
uv pip install -e .
```

Set 2 environment variables:

```
export LLM_API_KEY=your_api_key
export LLM_BASE_URL=https://your-endpoint.com (any OpenAI compatible)
```


# Running  with uv (recommended)


```
uv run soak run demo soak/data/yt-cfs.txt --output yt-cfs-example
```




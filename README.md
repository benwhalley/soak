
# soak: a DAG based method to describe text-analysis pipelines for qualitative/thematic analysis.

```
uv run soak run soak/pipelines/zspe.yaml \
   data/5LC.docx  \
   --output result_exercise  \
   -t short \
   -c excerpt_topics="Exercise and camaraderie" \
   && open result_exercise.html


uv run soak run soak/pipelines/zspe.yaml \
   data/5LC.docx  \
   --output result_fun  \
   -t short \
   -c excerpt_topics="Fun and family" \
   && open result_fun.html

```



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
export LLM_API_BASE=https://your-endpoint.com (any OpenAI compatible)
```


# Running  with uv (recommended)


```
uv run soak run demo soak/data/yt-cfs.txt --output yt-cfs-example
```




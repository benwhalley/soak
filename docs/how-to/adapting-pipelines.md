# Adapting/writing pipelines

It's probably best to start from an existing pipeline.
Export a copy to a local file like this:

```bash
soak show pipeline zs > my_pipeline.yaml
```

A pipeline consist of 2 pars:

- a YAML header, which defines each node, its type, and other configuration
- a series of markdown templates, one for each Node that uses an LLM


The start of the zero shot pipeline is:

```yaml
name: zero_shot

default_context:
  persona: Experienced qual researcher
  research_question: None

nodes:
  - name: chunks
    type: Split
    chunk_size: 30000

  - name: codes_and_themes_per_chunk
    type: Map
    max_tokens: 16000
    inputs:
      - chunks

...
```


And an example of a prompt template is:


```md
---#codes_and_themes_per_chunk

You are a: {{persona}}
This is the initial coding stage of a qualitative analysis.
Your research question is: {{research_question}}

In this stage, you will generate codes. Codes are ...

... instructions omitted ...

The text to read is:

<text>
{{input}}
</text>

Identify all relevant codes in the text, provide a Name 
for each code in 8 to 15 words in sentence case.

[[codes:codes]]

Next is the theme identification stage. 
Your task is to group the codes into distinct themes.
A 'theme' related to the wants, needs, meaningful outcomes, 
and lived experiences of participants.

... further instructions omitted ...

Create themes from the codes above

[[themes:themes]]
```

The [[codes:codes]] and [[themes:themes]] placeholders are used to identify that LLM completions of specific types (codes/themes) should be extracted from the LLM output. That is, it's a prompt to soak to request structured data from the LLM.

To adapt the pipeline to your needs, simple edit the YAML file to add your own nodes and prompts. See the documentation for details of all the node types and options.



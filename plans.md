look at the cli.py and think about how someone might use it. see models.py for the structure of DAGs when they run. Look at results2.json as an example of how output is saved.

I would like features to save the results of DAG runs into flat files ... i.e. to export the saved json into something someone can introspect the run.
  
This would be helpful to diagnose the prompts that actually ran (also see the internals of struckdown ChatterResult in ~/dev/struckown for details of how prompts and responses are stored). 

Also useful would be to see the output of each prompt, so for example when using a Map note, we would want to see the prompt and the response for each item. For a Reduce node, we'd want to see the output where all items have been concatenated together. For a Transform node, we'd want to see the input and output.

We could use folders to organise results for each node. Folders would be named after the node and node_type. Perhaps they could be numbered by the execution order? So a folder might be 01_map_NAMEOFNODE... Where the execution order has ties it's ok that the numbers are not unique.

Within each folder, we would save the prompt template, and each instantiation of the prompt template with data... all as separate files. So in a Map folder we might have:

- prompt_template.sd (struckdown format so sd)
- 01_prompt.md (markdown of the template with data from the first input item)
- 01_response.txt (a text version of the response from the first input item... i.e. what happens if get the .repsponse attribute of the ChatterResult object)
- 01_response.json (a json dump of the ChatterResult object in totality)
.. and so on for each input item

For a Transform node, we'd not need numbering, so just have the 
- prompt_template.sd
- prompt.md (the input)
- response.txt (the output)
- response.json (the ChatterResult object dumped)

For a Reduce node we'd just have:
- reduce_template.md
- reduced.txt (the concatenated outputs)


In the root folder of the output, we'd have a file called meta.txt which includes any runtime options used (e.g. context overrides, model name, etc.) and perhaps even the `soak run ...` command the user called.

small note: You might come across issues with the way that ChatterResult objects are serialised because they contain SegmentResult objects which have a field with Any type. Comment on whether this is problem and any easy fixes/


make a plan for this and a todolist which we can agree before you start work
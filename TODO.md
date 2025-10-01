

- classifier node for chunks
- quote validation



There is some weirdness with serialisatio of ChatterResult. It seems tobe related to ['response'] results being different if we dump to dict first/ The to_html method seems to rely on the object having been dumped to dict first which isn't ideal, but think it's becasue ChatterResult not reserialised properly so we adapted fucntionality around common case that it's already been serialised and reloaded.

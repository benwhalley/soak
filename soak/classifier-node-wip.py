import json
import logging
from pathlib import Path

import pandas as pd
from struckdown import LLM, chatter, chatter_async

logger = logging.getLogger("struckdown")
logger.setLevel(logging.DEBUG)


pmpt = """
This is text from a transcript of a patient who has recovered from CFS/ME.

We want to ask a number of questions about the transcript to categorise their background and experience.

Read it carefully and answer the questions below.

<transcript>

{{input}}

</transcript>

¡BEGIN

Did the patient suffer from CFS/ME or long covid?
Extract a quote or quotes from the transcript would help us make this classification.
[[cfs__evidence]]

Now pick an option:
[[pick:diagnosis|cfs,covid,both,none,unclear,no__evidence]]

¡OBLIVIATE


Is the patient now a professional helping other people recover from CFS/ME or long covid? Extract a quote or quotes from the transcript would help us make this classification.
[[is_professional__evidence]]

Now pick an option:
[[pick:is_professional|professional,not_a_professional,unclear,no__evidence]]


¡OBLIVIATE


Did the patient make a full or partial recovery?

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence".
[[recovery__evidence]]

Now pick an option:
[[pick:recovery|full,partial,unclear,no__evidence]]


¡OBLIVIATE

Was the patient bed-bound or very severely affected such that they could not leave their house?

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence".
[[bedbound__evidence]]

Now pick an option. Was the patient bed-bound or very severely affected such that they could not leave their house?
[[pick:bedbound|yes,no,unclear,no__evidence]]


¡OBLIVIATE

Was the patient suicidal at any point?

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence".
[[suicide__evidence]]

Now pick an option:
[[pick:suicide|yes,no,unclear,no__evidence]]


¡OBLIVIATE

Did the patient find advice and prescriptions from conventional medical practictioners helpful? By conventional medical practictioners, we mean doctors, nurses, and other healthcare professionals, or clinical psychologists working within conventional medical settings.

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence.
[[conventional__evidence]]

Now pick an option:
[[pick:conventional|helpful,unhelpful,mixed,unclear,no__evidence]]


¡OBLIVIATE

Did the patient find complementaty and alternative medicine helpful in their recovery? By CAM we use the NCCIH (US National Center for Complementary and Integrative Health) definition:

> “Complementary and alternative medicine (CAM) is a broad domain of healing resources that encompasses all health systems, modalities, and practices and their accompanying theories and beliefs, other than those intrinsic to the politically dominant health system of a particular society or culture in a given historical period.”

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence.
[[cam__evidence]]

Now pick an option:
[[pick:cam|helpful,unhelpful,mixed,unclear,no__evidence]]


¡OBLIVIATE

Did an individual professional help the patient recover from CFS/ME or long covid? For example a doctor, therapist, coach or some other person who is trained to deliver any kind of care.

Extract a quote or quotes from the transcript would help us make this classification. ONLY INCLUDE QUOTES FROM THE TEXT ABOVE. DO NOT FABRICATE QUOTES. If no relevant text is found, say "No evidence.
[[professional_helped__evidence]]

Now pick an option. Did an individual professional help the patient recover?
[[pick:professional_helped|yes,no,unclear,no__evidence]]

"""



df = pd.DataFrame([i.read_text() for i in list(Path("data/random10").glob("*.txt"))][:2])
df.columns = ["input"]


llm = LLM(model_name="litellm/gpt-4.1")



import asyncio

from tqdm.asyncio import tqdm_asyncio


async def process_decisions(df, pmpt, llm, max_concurrent=5):
    sem = asyncio.Semaphore(10)
    
    async def run_one(text):
        async with sem:
            return await chatter_async(
                pmpt, context={"input": text}, model=llm)
    
    tasks = [run_one(text) for text in df["input"]]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing transcripts")
    return results

results = asyncio.run(process_decisions(df, pmpt, llm))

output = pd.concat([df, pd.DataFrame([i.outputs for i in results])], axis=1)

output.to_csv("test.csv")





TASK FOR CLAUDE

the idea here is a new node that will apply a classifier prompt to the data and return the results as a dataframe or list of dicts with type Dic[str, Union[str, float, bool, int]]

The prompt will be something like the one above, and defined in a node like others.
the input can be anything. we could use a Split note prior to the ClassifierNode an it would then use that as the input.__annotations__


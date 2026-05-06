"""One-shot generator for ``codes_long.json`` (long-context probe fixture).

Run once and commit the output JSON. The script seeds five conceptual
clusters about remote work and emits 50 ``Code`` objects with 2-3 quotes
each. Hashes are computed by ``Code.hash()`` and ``Quote.hash()`` so the
fixture re-validates cleanly via ``Code(**dict)``.

Usage::

    uv run python -m soak.evals.data._generate_codes_long > soak/evals/data/codes_long.json

The same content is intentionally synthetic-but-coherent so that strong
models can still produce sensible consolidations / themes; the probe
metric is hash-fidelity, not consolidation quality.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List

from soak.models.base import Code, Quote


CLUSTERS: Dict[str, List[Dict[str, object]]] = {
    "Home Environment": [
        {
            "name": "Improvised Workspace",
            "description": "Participants describe setting up makeshift desks at kitchen tables, bedrooms, or sofas, often with poor ergonomics and inadequate equipment.",
            "quotes": [
                "I was working off the kitchen table for the first six months, and my back paid for it.",
                "My 'office' was a folding chair and a laptop on top of a moving box.",
            ],
        },
        {
            "name": "Investing in the Home Office",
            "description": "Over time, participants gradually invested in proper chairs, monitors, and lighting to make remote work sustainable.",
            "quotes": [
                "Once I bought a real chair and a second monitor I realised how much time I had been wasting hunched over.",
                "The webcam ring light made me look human on calls again.",
            ],
        },
        {
            "name": "Background Noise and Interruptions",
            "description": "Children, neighbours, building work, and pets created constant interruptions that participants had to learn to manage during calls.",
            "quotes": [
                "The dog barks every time the postman comes, which is roughly once per important meeting.",
                "I had to put a sign on the door because my kids would wander in mid-call.",
            ],
        },
        {
            "name": "Lighting and Camera Anxiety",
            "description": "Participants reflected on how awareness of their on-camera appearance shaped their workspace setup and self-presentation.",
            "quotes": [
                "I learned the hard way that morning sun makes me look like a ghost on Zoom.",
                "I started turning my camera off because I couldn't bear seeing my own face all day.",
            ],
        },
        {
            "name": "Bleeding into Living Spaces",
            "description": "Several participants noted that work equipment increasingly took over kitchens, bedrooms, and other shared rooms in the home.",
            "quotes": [
                "Eventually the laptop just lived on the coffee table and we ate around it.",
                "There was no clear edge anymore between the bedroom and the office.",
            ],
        },
        {
            "name": "Heating and Energy Costs",
            "description": "Participants raised practical concerns about energy bills rising as they spent full days at home with heating and computers running.",
            "quotes": [
                "My winter heating bill basically doubled the year I started working from home.",
                "I started wearing a hat indoors before I'd touch the thermostat.",
            ],
        },
        {
            "name": "Pets at the Desk",
            "description": "Pets featured frequently as both a comfort and a source of distraction throughout the working day.",
            "quotes": [
                "My cat sleeps on the keyboard for at least three hours of every day.",
                "Honestly the dog has been the best colleague I've had in years.",
            ],
        },
        {
            "name": "Boundary Marking with Family",
            "description": "Participants developed deliberate rituals -- closed doors, signs, headphones -- to signal to family members that they were 'at work'.",
            "quotes": [
                "Headphones on means do not approach. Headphones off, fair game.",
                "We had to agree as a household that the spare room door being shut was sacred.",
            ],
        },
        {
            "name": "Window and Natural Light",
            "description": "Access (or lack of access) to natural light at the desk was repeatedly cited as central to mood and wellbeing.",
            "quotes": [
                "My old cubicle had no window at all -- now I face a garden and it's transformed how I feel by lunch.",
                "The flat is dark and that's been harder than I expected on grey days.",
            ],
        },
        {
            "name": "Sound Insulation Hacks",
            "description": "Participants described creative low-cost solutions -- duvets over doors, foam panels, closets -- to reduce echo and outside noise on calls.",
            "quotes": [
                "I record podcasts inside a wardrobe full of coats, and it actually sounds great.",
                "We taped sound foam to the back of the door because the alley behind was too loud.",
            ],
        },
    ],
    "Communication and Collaboration": [
        {
            "name": "Asynchronous Habits",
            "description": "Teams shifted to written, asynchronous communication for many tasks that would previously have been a quick desk conversation.",
            "quotes": [
                "We try to write everything down now -- if it isn't in Slack, it didn't happen.",
                "I send a Loom instead of booking a meeting and it's been a real unlock.",
            ],
        },
        {
            "name": "Meeting Overload",
            "description": "Participants reported a steady creep of back-to-back video meetings replacing what used to be informal coffee chats.",
            "quotes": [
                "On a bad day I'm in seven hours of video calls and have done no actual work.",
                "Every chat that used to be a corridor moment is now a calendar invite.",
            ],
        },
        {
            "name": "Loss of Hallway Conversations",
            "description": "The lack of corridor and lunch interactions was named as a major loss for organisational learning, gossip, and casual problem solving.",
            "quotes": [
                "I used to learn most of my job by overhearing things, and that's just gone.",
                "I miss bumping into a senior person at the coffee machine.",
            ],
        },
        {
            "name": "Tooling Fragmentation",
            "description": "Multiple competing tools (Slack, Teams, Notion, email, Jira) created friction and unclear conventions about where things belonged.",
            "quotes": [
                "I have the same conversation in four tools because no-one agreed where it should live.",
                "We have a wiki and a Slack and a Notion and a shared drive, and the truth is in none of them.",
            ],
        },
        {
            "name": "Camera-Off Culture",
            "description": "Norms around whether cameras should be on varied widely and were a frequent point of friction or relief for participants.",
            "quotes": [
                "Our team agreed cameras off by default and the meetings got 30% shorter.",
                "When my manager has the camera off, I can't read the room at all.",
            ],
        },
        {
            "name": "Onboarding New Joiners",
            "description": "Bringing new hires into a remote team without informal interactions was widely regarded as harder and slower than in-person onboarding.",
            "quotes": [
                "It took my new hire about three months to understand the team's actual dynamics.",
                "We tried to overcompensate with welcome calls and it still didn't fix it.",
            ],
        },
        {
            "name": "Time Zone Coordination",
            "description": "Distributed teams reported significant overhead in finding meeting times and the friction of waiting overnight for replies.",
            "quotes": [
                "Half my morning is gone before any of my US colleagues are even online.",
                "We do a lot of work by handoff and trust now -- there's no other way.",
            ],
        },
        {
            "name": "Documenting Decisions",
            "description": "Teams developed stronger norms for writing down decisions because hallway memory was no longer reliable.",
            "quotes": [
                "If a decision isn't in the doc with someone's name on it, it didn't happen.",
                "We learned the hard way that 'we agreed in that call' is not a record.",
            ],
        },
        {
            "name": "Virtual Whiteboarding",
            "description": "Tools like Miro and FigJam were appreciated but participants noted that real-time collaboration on them rarely matched in-person sessions.",
            "quotes": [
                "Miro is fine but two people moving stickies at once on Zoom is a mess.",
                "I miss standing at a whiteboard with markers in my hand.",
            ],
        },
        {
            "name": "Manager Visibility Anxiety",
            "description": "Participants described pressure -- real or perceived -- to be constantly responsive on chat to demonstrate that they were working.",
            "quotes": [
                "I keep Slack on because I worry about looking idle.",
                "There's an unspoken rule that you reply within five minutes or someone is annoyed.",
            ],
        },
    ],
    "Work-Life Balance": [
        {
            "name": "Eroding Boundaries",
            "description": "Participants described work blurring into evenings and weekends as the physical separation of office from home disappeared.",
            "quotes": [
                "I'd answer one Slack message at 9pm and look up at midnight.",
                "There's no commute that ends the day for me anymore.",
            ],
        },
        {
            "name": "Reclaimed Commute Time",
            "description": "Many participants framed the absence of a daily commute as the single biggest improvement in their lives.",
            "quotes": [
                "I got two hours a day back -- that's a different life, frankly.",
                "I haven't missed standing on a crowded train for a single second.",
            ],
        },
        {
            "name": "Lunch as a Chore",
            "description": "Lunch shifted from a social or restorative pause to something brief and functional eaten at the desk.",
            "quotes": [
                "Lunch is a sandwich at the keyboard now, if I remember at all.",
                "I used to eat with colleagues and now I eat with my laptop.",
            ],
        },
        {
            "name": "Always-on Pressure",
            "description": "The expectation -- explicit or implicit -- that one would be reachable across longer hours was widely felt.",
            "quotes": [
                "Even on holiday I find myself opening email 'just to keep on top of things'.",
                "My boss never says it directly but the late-night messages tell you what's expected.",
            ],
        },
        {
            "name": "Family Logistics Rebalance",
            "description": "Participants with children renegotiated the day around school runs, naps, and after-school care alongside their work.",
            "quotes": [
                "I work an hour before they wake up and finish after bedtime -- the middle is a circus.",
                "We split who does the school run by who has fewer 9 a.m. calls.",
            ],
        },
        {
            "name": "Exercise Becoming Possible",
            "description": "Several participants reported that the time savings from no commute let them build new exercise routines.",
            "quotes": [
                "I run three mornings a week now because I have the daylight.",
                "I cycle on my lunch break and feel like a different person at 2pm.",
            ],
        },
        {
            "name": "Weekend Work Drift",
            "description": "Participants noted a quiet expansion of work into weekends, especially Sundays, often described as 'just catching up'.",
            "quotes": [
                "Sunday evening is its own little working day and I hate that.",
                "I tell myself I'll just clear the inbox and then it's two hours gone.",
            ],
        },
        {
            "name": "Small Domestic Wins",
            "description": "Routine domestic tasks -- laundry, post collection, deliveries -- became markedly easier to fit into the working day.",
            "quotes": [
                "I can put the laundry on between meetings and it's revolutionary.",
                "Being home for a parcel without taking a half day is genuinely liberating.",
            ],
        },
        {
            "name": "Vacation Guilt",
            "description": "Despite encouragement to rest, participants described feeling pressure to stay reachable while on annual leave.",
            "quotes": [
                "I checked email on a Greek beach last week and I wasn't proud of it.",
                "It feels harder to truly switch off than it did from an office.",
            ],
        },
        {
            "name": "End-of-Day Rituals",
            "description": "Some participants invented deliberate rituals -- shutting the laptop in a drawer, a short walk, a different jumper -- to mark the end of the working day.",
            "quotes": [
                "I take a walk around the block at 6pm to fake the commute.",
                "Closing the laptop and putting it under the bed sounds silly but it works.",
            ],
        },
    ],
    "Mental Health and Isolation": [
        {
            "name": "Daily Loneliness",
            "description": "Participants described a chronic, low-grade loneliness from spending most of the working day alone, even when otherwise content with remote work.",
            "quotes": [
                "Some days I realise I haven't spoken aloud to another human until 5pm.",
                "Loneliness is not the right word but something close to it.",
            ],
        },
        {
            "name": "Therapist Recommendations",
            "description": "Several participants reported that mental-health professionals encouraged them to reintroduce social structure and routine into remote-working life.",
            "quotes": [
                "My therapist basically told me I needed to build small social rituals back in.",
                "Even my GP suggested I try a co-working day a week.",
            ],
        },
        {
            "name": "Identity Beyond the Job",
            "description": "Without the social cues of an office, participants described having to actively construct identities outside of work.",
            "quotes": [
                "I used to be 'the person at the office' and now I have to figure out who I am the rest of the time.",
                "Hobbies stopped being optional once the job stopped being a place.",
            ],
        },
        {
            "name": "Mood Tracking the Weather",
            "description": "Several participants noticed that their mood became more sensitive to weather, daylight, and seasons because they no longer left the house for work.",
            "quotes": [
                "On a grey week I can feel myself drifting and I have to do something about it.",
                "Sun in the kitchen at 10am makes my whole day better.",
            ],
        },
        {
            "name": "Substance Creep",
            "description": "Some participants admitted that alcohol consumption or snacking habits drifted upward without the structure and observation of an office environment.",
            "quotes": [
                "There's no commute home so the wine after work happens half an hour earlier.",
                "Nobody can see me eat biscuits all day, and apparently that's all I needed to know.",
            ],
        },
        {
            "name": "Body Movement Decline",
            "description": "Participants noted a sharp drop in incidental walking and standing compared to office life.",
            "quotes": [
                "Five hundred steps in a whole working day is a normal number for me now.",
                "Even walking to the printer used to add up.",
            ],
        },
        {
            "name": "Anxiety About Re-emergence",
            "description": "Returning to in-person work or social events became a source of anxiety for participants who had adjusted deeply to home life.",
            "quotes": [
                "The first time I had to go back to the office I felt physically sick.",
                "Big events feel louder and harder than they used to.",
            ],
        },
        {
            "name": "Need for Human Voices",
            "description": "Participants described seeking out radio, podcasts, or background calls during the day to compensate for the absence of voices in the home.",
            "quotes": [
                "I leave the radio on for company even when I'm not listening.",
                "Podcasts have become my ambient colleagues.",
            ],
        },
        {
            "name": "Erratic Sleep Patterns",
            "description": "Without a fixed start time, participants reported sleep schedules that drifted later, with knock-on effects on energy and focus.",
            "quotes": [
                "I don't have to be anywhere by nine, so I drift later and later until I have to reset.",
                "My weekends and weekdays look the same now and I think that's part of the problem.",
            ],
        },
        {
            "name": "Coffee Shop as Refuge",
            "description": "Many participants began routinely working from cafes specifically for the soft social presence of strangers.",
            "quotes": [
                "I go to the cafe just to work near people, even if I never speak to them.",
                "Hearing other lives at neighbouring tables is somehow grounding.",
            ],
        },
    ],
    "Productivity and Motivation": [
        {
            "name": "Deep Focus Gains",
            "description": "Some participants reported significant gains in concentration on solo, deep work that had been difficult in open-plan offices.",
            "quotes": [
                "I get more done in two morning hours at home than a whole afternoon in the office.",
                "I can think a thought all the way through now without being interrupted.",
            ],
        },
        {
            "name": "Procrastination Patterns",
            "description": "Without ambient social pressure, participants described characteristic procrastination patterns -- laundry, the kettle, news sites -- they had to learn to manage.",
            "quotes": [
                "I have a whole choreography of small jobs that aren't work but feel productive.",
                "The kettle gets boiled an unholy number of times before I open the doc.",
            ],
        },
        {
            "name": "Calendar as Discipline",
            "description": "Time-blocking and explicit calendars became important tools for participants trying to keep momentum across an unstructured day.",
            "quotes": [
                "If it isn't on the calendar with a colour, it doesn't happen.",
                "I block 9-11 every day for the work that actually matters.",
            ],
        },
        {
            "name": "Energy Audit Awareness",
            "description": "Participants developed sharper awareness of their own energy peaks and troughs and matched task type to time of day accordingly.",
            "quotes": [
                "Mornings for thinking, afternoons for meetings -- I learned that the hard way.",
                "I try not to put deep work after 2pm because I'll just stare at it.",
            ],
        },
        {
            "name": "Distraction Engineering",
            "description": "Strategies like phone in another room, website blockers, focus modes, and turned-off notifications were widely adopted.",
            "quotes": [
                "I leave the phone in a different room and my output doubled.",
                "Once I muted Slack for 90 minutes I realised how much it had been pulling at me.",
            ],
        },
        {
            "name": "Creative Output Variability",
            "description": "Creative quality felt more variable to many participants -- great days were better, bad days were worse, with fewer mid-range days.",
            "quotes": [
                "On a good day I write the best thing I've written in years; on a bad day, nothing.",
                "There's less of the medium-okay day that the office used to fill out.",
            ],
        },
        {
            "name": "Pomodoro and Sprint Methods",
            "description": "Time-bounded work techniques were specifically credited with rescuing motivation on hard or boring tasks.",
            "quotes": [
                "Twenty-five minutes is the longest I can promise myself, and that's enough.",
                "I do a sprint with a colleague on a video call for hard tasks and it just works.",
            ],
        },
        {
            "name": "Avoiding Drift",
            "description": "Participants described needing explicit re-orientation every morning to avoid drifting into low-value tasks.",
            "quotes": [
                "If I don't write the day's three priorities down before email, the day is lost.",
                "Without a plan I just pick up whatever is loudest and that's not real work.",
            ],
        },
        {
            "name": "Reward and Closure Habits",
            "description": "Small rewards and explicit closure rituals -- crossing things off, a walk after a hard meeting -- helped participants maintain morale.",
            "quotes": [
                "I close every day by writing what I finished, even on bad days.",
                "A walk after a hard meeting is non-negotiable now.",
            ],
        },
        {
            "name": "Loss of External Validation",
            "description": "Without immediate feedback from a manager or colleagues, some participants lost a sense of whether they were doing well.",
            "quotes": [
                "Nobody walks past my desk and goes 'nice'. I have to invent that for myself.",
                "I miss the unscripted feedback you'd get just by being seen working.",
            ],
        },
    ],
}


def main():
    rng = random.Random(1)  # deterministic
    out = []
    for cluster, items in CLUSTERS.items():
        for entry in items:
            quotes = []
            for q_text in entry["quotes"]:
                source_id = f"interview_{rng.randint(1, 12):02d}"
                q = Quote(text=q_text, source=source_id)
                quotes.append({
                    "type": "quote",
                    "text": q.text,
                    "source": q.source,
                })
            code_obj = Code(
                name=entry["name"],
                description=entry["description"],
                quotes=quotes,
            )
            d = code_obj.model_dump(mode="json")
            d.pop("llm_config", None)
            d.pop("slug", None)
            d.pop("resolved_quotes", None)
            d.pop("theme_name", None)
            out.append(d)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

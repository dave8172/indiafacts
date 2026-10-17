# Contributing

Pull requests are welcome. These rules keep the tool what it says it is.

**Adding a source** (`sources.json`):
- Give its rights basis in India: author dead more than 60 years, government work more than 60 years old, or
  an Act, Gazette notice, judgment or commission report laid before a legislature. If you can't state one,
  the source can't go in, however good it is.
- Never add a copy someone uploaded without the right to share it.
- Say what kind of evidence it is (`standing`) and keep its `note` to plain facts: what it is, who wrote it,
  when. No description of whether it is reliable, biased or important.
- Prefer sources that add another side's record of the same events.

**Adding a website** (`sites.json`): grade it by what it is the original of (A issuer, B faithful copy,
C research, D news and open-edit sites, never used as a source). A grade says nothing about whether the
content is true.

**Changing prompts or retrieval:** run the evals (`evals/run.py`) before and after. A change that makes the
two sides of a mirrored pair less alike is a regression, even if other numbers improve.

**Never:** add memory of users, rank or filter by anyone's views, use news or Wikipedia as a source of facts,
or let an answer judge a person, party, government or event.

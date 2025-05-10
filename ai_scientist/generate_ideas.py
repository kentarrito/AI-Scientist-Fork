import json
import os
import os.path as osp
import time
import random
from typing import List, Dict, Union

import backoff
import requests

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers, create_client, AVAILABLE_LLMS, extract_text_inside_backticks

S2_API_KEY = os.getenv("S2_API_KEY")

brainstorming_system_msg = """You are super genius and try to brainstorm to make a novel idea on the task below. Brainstorm and think of many thoughts following the user's insturuction.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

brainstorming_prompt = """

Here are the brainstorming that you have already made before:

'''
{brainstorming_history}
'''

Try to expand your thoughts and question about this topic following the agents below. 
{agents}

Respond in the following format:

BRAINSTORMING:
```text
<TEXT>
```

In <TEXT>, try to expand your thoughts about the topic.
"""

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here is some additional brainstorming to guide your creativity:

'''
{brainstorming}
'''

Using the above brainstorming as inspiration, come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided. Draw clear connections between your idea and the brainstormed insights where possible.  
Note that you will not have access to any additional resources or datasets. Make sure any idea is not overfit to the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments. Justify how the idea is different from the existing ones.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


# GENERATE IDEAS
def generate_ideas_with_brainstorming(
        base_dir,
        agents,
        brainstorming_history,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    print()
    msg_history = []
    bs_msg_history = []
    print("Brainstorming...")
    chosen_agents = random.sample(agents, 3)
    for i_bs in range(3):
        text, bs_msg_history = get_response_from_llm(
            chosen_agents[i_bs],
            client=client,
            model=model,
            system_message=brainstorming_system_msg.format(
                task_description=prompt["task_description"],
                code=code,
            ),
            msg_history=bs_msg_history,
        )
    print("bs_msg_history:")
    print(bs_msg_history)
    print()

    for _ in range(max_num_generations):
        
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)
            
            print(f"Generating idea {_ + 1}/{max_num_generations} ...")
            text, msg_history = get_response_from_llm(
                idea_first_prompt.format(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    brainstorming=bs_msg_history,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
            )
            ## PARSE OUTPUT
            json_output = extract_json_between_markers(text)
            assert json_output is not None, "Failed to extract JSON from LLM output"
            print()
            print(f"Iteration 1/{num_reflections} Generated Ideas: ")
            print(json_output)

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    text, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                    )
                    ## PARSE OUTPUT
                    json_output = extract_json_between_markers(text)
                    assert (
                            json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    print()
                    print(f"Iteration {j + 2}/{num_reflections} Generated Ideas: ")
                    print(json_output)

                    if "I am done" in text:
                        print()
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas, bs_msg_history


# GENERATE IDEAS
def generate_bs_agents_dataset(
        base_dir,
        agents,
        client,
        model,
        skip_generation=False,
        max_num_generations=20,
        num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    print()
    print("Brainstorming...")
    
    bs_msg_histories = {}  # {(depth,branch):[{"system":}...], ...}
    bs_agent_id_histories = {}  # {(depth,branch):id, ...}
    all_ideas = {}

    num_depth = 3
    num_branch = 2
    n_agents = len(agents)
    bs_agent_tree = {}  # [{"agent_id":, "node_id":[0], "bs_msg":[{"role":"system", "content":"..."}], "ideas":[{}], "children":[{"msg":[{"role":"user"}, {"role":"assistant"}]}]}, ]

    def build_bs_agent_tree(agents, *, num_depth=3, num_branch=2, seed=None):
        if seed is not None:
            random.seed(seed)
        if not agents:
            raise ValueError("agents list is empty.")

        n_agents = len(agents)

        def _grow(node, depth, forbidden):
            if depth == num_depth:
                return
            available = list(set(range(n_agents)) - forbidden)
            if not available:
                raise RuntimeError("Not enough distinct agents for the depth requested.")

            picked_a_idxs = random.sample(available, num_branch)
            for b in range(num_branch):
                a_idx = picked_a_idxs[b]
                child = {
                    "agent"   : agents[a_idx],
                    "agent_ids": node["agent_ids"] + [a_idx],
                    "agent_id": a_idx,
                    "node_ids" : node["node_ids"] + [b],
                    "bs_msg"  : [],
                    "ideas"   : [],
                    "children": [],
                }
                node["children"].append(child)
                _grow(child, depth + 1, forbidden | {a_idx})

        root = {"agent_id": None, "agent_ids": [], "node_ids": [], "bs_msg": [], "ideas": [], "children": []}
        _grow(root, 0, set())
        return root

    def get_assistant_msg(
        node,
        *,                          # keyword-only
        history_so_far,             # ancestor conversation
        prompt, code,
        client, model,
    ):
        """
        Returns a *new* history list that is `history_so_far`
        plus ONE user/assistant pair for this depth,
        and the parsed idea dict produced in that exchange.
        """
        if node["agent_ids"]==[]: agent_id = None
        else: agent_id = node["agent_ids"][-1]

        # ------------------------------------------------------------------ copy
        bs_msg = list(history_so_far)               # preserves ancestor msgs

        # ---------------------------------------------------- inject system once
        if not any(m["role"] == "system" for m in bs_msg):
            bs_msg.append({
                "role": "system",
                "content": brainstorming_system_msg.format(
                    task_description = prompt["task_description"],
                    code             = code,
                )
            })

        # --------------------------------------------------- construct user turn
        user_prompt = f"[Agent: {agent_id}] " + idea_first_prompt.format(
            task_description = prompt["task_description"],
            code             = code,
            brainstorming    = bs_msg,
            num_reflections  = num_reflections,
        )

        # talk to LLM *once* (temp_history is a throw-away list)
        temp_history = []
        assistant_txt, temp_history = get_response_from_llm(
            agent          = agent_id,
            client         = client,
            model          = model,
            system_message = idea_system_prompt,
            msg_history    = temp_history,
            user_message   = user_prompt,
        )

        # append exactly one pair to the running history
        bs_msg.append({"role": "user",      "content": user_prompt})
        bs_msg.append({"role": "assistant", "content": assistant_txt})

        # parse idea
        idea_json = extract_json_between_markers(assistant_txt) or {"idea": assistant_txt,
                                                                    "agent": agents[agent_id]}
        return bs_msg, [idea_json]


    def populate_tree(node, history_so_far, **llm_kwargs):
        """
        Depth-first traversal.
        history_so_far already obeys the 1 + depth*2 rule.
        """
        if node["agent_id"] is not None:         # skip dummy root
            node["bs_msg"], node["ideas"] = get_assistant_msg(
                node,
                history_so_far = history_so_far,
                **llm_kwargs,
            )
            next_history = node["bs_msg"]
        else:
            next_history = history_so_far

        for child in node["children"]:
            populate_tree(child, next_history, **llm_kwargs)

    bs_agent_tree = build_bs_agent_tree(
        agents, num_depth=num_depth, num_branch=num_branch, seed=42
    )

    populate_tree(
        bs_agent_tree,
        history_so_far = [],      # start empty
        prompt         = prompt,
        code           = code,
        client         = client,
        model          = model,
    )

    ## SAVE IDEAS

    with open(osp.join(base_dir, "bs_agent_tree.json"), "w") as f:
        json.dump(bs_agent_tree, f, indent=4)

    return bs_agent_tree


# GENERATE IDEAS OPEN-ENDED
def generate_next_idea(
    base_dir,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "experiment.py"), "r") as f:
            code = f.read()
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                text, msg_history = get_response_from_llm(
                    idea_first_prompt.format(
                        task_description=prompt["task_description"],
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    )
                    + """
Completed ideas have an additional "Score" field which indicates the assessment by an expert ML reviewer.
This is on a standard 1-10 ML conference scale.
Scores of 0 indicate the idea failed either during experimentation, writeup or reviewing.
""",
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                )
                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"
                print(json_output)

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        text, msg_history = get_response_from_llm(
                            idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                        )
                        ## PARSE OUTPUT
                        json_output = extract_json_between_markers(text)
                        assert (
                                json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        print(json_output)

                        if "I am done" in text:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(query, result_limit=10, engine="semanticscholar") -> Union[None, List[Dict]]:
    if not query:
        return None
    if engine == "semanticscholar":
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(
            f"Response Content: {rsp.text[:500]}"
        )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)
        if not total:
            return None

        papers = results["data"]
        return papers
    elif engine == "openalex":
        import pyalex
        from pyalex import Work, Works
        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print("[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!")
        else:
            pyalex.config.email = mail

        def extract_info_from_work(work: Work, max_abstract_length: int = 1000) -> dict[str, str]:
            # "Unknown" is returned when venue is unknown...
            venue = "Unknown"
            for i, location in enumerate(work["locations"]):
                if location["source"] is not None:
                    venue = location["source"]["display_name"]
                    if venue != "":
                        break
            title = work["title"]
            abstract = work["abstract"]
            if abstract is None:
                abstract = ""
            if len(abstract) > max_abstract_length:
                # To avoid context length exceed error.
                print(f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars.")
                abstract = abstract[:max_abstract_length]
            authors_list = [author["author"]["display_name"] for author in work["authorships"]]
            authors = " and ".join(authors_list) if len(authors_list) < 20 else f"{authors_list[0]} et al."
            paper = dict(
                title=title,
                authors=authors,
                venue=venue,
                year=work["publication_year"],
                abstract=abstract,
                citationCount=work["cited_by_count"],
            )
            return paper

        works: List[Dict] = Works().search(query).get(per_page=result_limit)
        papers: List[Dict[str, str]] = [extract_info_from_work(work) for work in works]
        return papers
    else:
        raise NotImplementedError(f"{engine=} not supported!")



novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{task_description}
<experiment.py>
{code}
</experiment.py>
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
```thought
<THOUGHT>
```

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''



make_agent_system_msg = """You are a meticulous AI researcher (PhD level) tasked with revitalizing ideas that failed the novelty check. The literature survey is complete—you know this idea overlaps existing work. Your mission is to analyze why the idea lacked novelty, suggest concrete improvements, and define brainstorming sentences to explore promising new directions. Respond your answer following the instructions."""

make_agent_prompt = '''## Provided Context

### Task Description
{task_description}

### Experiment Harness
```python
# experiment harness
<experiment.py>
{code}
</experiment.py>
```

### Idea from Round {current_round} of {num_rounds}
"""
{idea}
"""

### Relevant papers
"""
{last_query_results}
"""

### Previous Analysis
- THOUGHT:
  """{thought}"""
- NOVELTY JUDGMENT:
  """{novelty}"""

## Instructions

1. **Think Why Idea was not Novel**: Refering to relevant papers which are already published, think why the idea was not novel. 
2. **Think How the Idea Can Be More Novel**: Imagine as many thinking processes as possible which may have led to more novel idea. For example, "Thinking about why the issue occured might have led to an idea which tackled more general and bigger problem in this field".
3. **Give Short and Abstract Brainstorming Sentences**: Expanding your imagination, design 3 to 5 brainstorming sentences that may lead more novel idea. All the setences should be abstract and general so that it can be applied to other papers too. These are the examples of the instructions; “Explore all the possibilities of other ideas”, “Check if there are enough conditions to solve an issue”, “Imagine what condition will lead you to solve the issue”
4. **Generate Output**: Based on the result so far, answer your output enclosing it in a JSON code block like output format below

```json
{{
  "brainstormings": [
    "<BrainstormingSentence>",
    ...
  ]
}}
```

Let's think step by step following each step of the instructions.'''

'''
def check_idea_novelty(
        ideas,
        base_dir,
        client,
        model,
        max_num_iterations=10,
        engine="semanticscholar",
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                thought_output = extract_text_inside_backticks(text, "thought")
                #response_output = extract_text_inside_backticks(text, "response")
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10, engine=engine)
                if papers is None:
                    papers_str = "No papers found."

                paper_strings = []
                for i, paper in enumerate(papers):
                    paper_strings.append(
                        """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                            i=i,
                            title=paper["title"],
                            authors=paper["authors"],
                            venue=paper["venue"],
                            year=paper["year"],
                            cites=paper["citationCount"],
                            abstract=paper["abstract"],
                        )
                    )
                papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(base_dir, "ideas.json")
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


def check_idea_novelty_and_make_agents(
        ideas,
        base_dir,
        client,
        model,
        n_bs_step,
        max_num_iterations=10,
        engine="semanticscholar",
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    n_ideas = len(ideas)
    n_novels = 0
    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""
        thought_output = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                thought_output = extract_text_inside_backticks(text, "thought") or thought_output

                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                # parse JSON
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                # search
                query = json_output.get("Query", "")
                papers = search_for_papers(query, result_limit=10, engine=engine)
                if not papers:
                    papers_str = "No papers found."
                else:
                    paper_strings = []
                    for i, paper in enumerate(papers):
                        paper_strings.append(
                                """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                    i=i,
                                    title=paper["title"],
                                    authors=paper["authors"],
                                    venue=paper["venue"],
                                    year=paper["year"],
                                    cites=paper["citationCount"],
                                    abstract=paper["abstract"],
                                )
                        )
                    papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel
        if novel: n_novels += 1

        print()
        print(f"novelty: {novel}")

        # If not novel, generate revitalization Agents
        if not novel:
            print()
            print(f"Generating agents to revitalize idea {idx}: {idea['Name']}")
            msg_hist_agents = []
            system_msg = make_agent_system_msg
            user_prompt = make_agent_prompt.format(
                task_description=task_description,
                code=code,
                current_round=1,
                num_rounds=1,
                idea=json.dumps(idea),
                last_query_results=papers_str,
                thought=thought_output,
                novelty="not novel"
            )
            agent_text, _ = get_response_from_llm(
                user_prompt,
                client=client,
                model=model,
                system_message=system_msg,
                msg_history=msg_hist_agents
            )

            agents_json = json.loads(agent_text)
            agents = agents_json.get("brainstormings")
            #agents_json = extract_json_between_markers(agent_text)
            idea["bs_agents"] = agents_json.get("brainstormings") if isinstance(agents_json, dict) else None

            print()
            print(f"Generated agents: {idea['bs_agents']}")

        else:
            agents = []

    print()
    print()
    print("Novelty check Finished")
    print(f"Brainstorming Step: {n_bs_step}")
    print(f"{n_novels} / {n_ideas} Novel")
    print()
    # save back
    results_file = osp.join(base_dir, "novelty_result.json")
    print()
    print("results_file: ", results_file)
    if os.path.exists(results_file):
        with open(results_file) as f:
            novelty_result = json.load(f)
    else:
        novelty_result = {}
    novelty_result[f"Brainstorming Step {n_bs_step}"] = {"n_novels":n_novels, "n_ideas":n_ideas}
    with open(results_file, "w") as f:
        json.dump(novelty_result, f, indent=4)

    results_file = osp.join(base_dir, "bs_prompts.json")
    bs_prompts = {"brainstorming_prompt":brainstorming_prompt, "idea_first_prompt":idea_first_prompt, "make_agent_system_msg":make_agent_system_msg, "make_agent_prompt":make_agent_prompt}
    with open(results_file, "w") as f:
        json.dump(bs_prompts, f, indent=4)
    print()
    print("File Saved")

    return ideas, agents

'''

def check_idea_novelty_in_bs_agent_tree(
    bs_agent_tree,  # {"agent_ids":[],"bs_msg":[],"children":[{same structure}, ]}
    base_dir,
    client,
    model,
    n_bs_step,
    max_num_iterations=10,
    engine="semanticscholar",
):
    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        task_description = prompt["task_description"]

    def check(ideas):
        idea = ideas[0]

        print(f"\nChecking novelty of idea: {idea['Name']}")
        
        novel = False
        msg_history = []
        papers_str = ""
        thought_output = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        task_description=task_description,
                        code=code,
                    ),
                    msg_history=msg_history,
                )
                thought_output = extract_text_inside_backticks(text, "thought") or thought_output

                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                # parse JSON
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                # search
                query = json_output.get("Query", "")
                papers = search_for_papers(query, result_limit=10, engine=engine)
                if not papers:
                    papers_str = "No papers found."
                else:
                    paper_strings = []
                    for i, paper in enumerate(papers):
                        paper_strings.append(
                                """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                    i=i,
                                    title=paper["title"],
                                    authors=paper["authors"],
                                    venue=paper["venue"],
                                    year=paper["year"],
                                    cites=paper["citationCount"],
                                    abstract=paper["abstract"],
                                )
                        )
                    papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        print()
        print(f"novelty: {novel}")

        return [novel]


    def add_novelty_to_tree(node, depth):

        if node["children"] == []:
            return None

        else:
            for i, node_dict in enumerate(node["children"]):
                novelties = check(node_dict["ideas"])
                node["children"][i]["novelties"] = novelties

                add_novelty_to_tree(node_dict, depth + 1)
            

    #bs_agent_tree = {"agent_id": None, "agent_ids": [], "node_ids": [], "bs_msg": [], "ideas": [], "children": []}
    add_novelty_to_tree(bs_agent_tree, 0)

    return bs_agent_tree


if __name__ == "__main__":
    MAX_NUM_GENERATIONS = 32
    NUM_REFLECTIONS = 5
    import argparse

    parser = argparse.ArgumentParser(description="Generate AI scientist ideas")
    # add type of experiment (nanoGPT, Boston, etc.)
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip idea generation and use existing ideas.",
    )
    parser.add_argument(
        "--check-novelty",
        action="store_true",
        help="Check novelty of ideas.",
    )
    args = parser.parse_args()

    # Create client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    results_dir = osp.join("results", args.experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=MAX_NUM_GENERATIONS,
        num_reflections=NUM_REFLECTIONS,
    )
    if args.check_novelty:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
        )

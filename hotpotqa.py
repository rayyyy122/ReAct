# Set Up
import os
import re
import openai
import random
import time
 
openai.api_key = os.environ["OPENAI_API_KEY"]

'''
def llm(prompt, stop=["\n"]):
    response = openai.Completion.create(
      #model="text-davinci-002",
      model="gpt-4o-mini",
      prompt=prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response["choices"][0]["text"]
'''

def llm(prompt, stop=["\n"], max_retries=10):
    """Use completion API instead of chat - closer to original paper"""
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  # Completion model!
                prompt=prompt,
                temperature=0,
                max_tokens=100,
                stop=stop
            )
            return response["choices"][0]["text"]
        except Exception as e:
            time.sleep(5 * (2 ** attempt))
            if attempt == max_retries - 1:
                raise e

import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# ReAct
import json
import sys

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    
    prompt += question + "\n"
    
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        
        # 1. STOP SEQUENCES: 
        # Removed generic "\nObservation" because it cuts off thoughts like 
        # "My observation is that..."
        stop_sequences = [
            f"\nObservation {i}:", 
            f"\nObservation {i}",
            f"\nObservation:"
        ]
        
        # 2. PROMPT FIX: 
        # Removed the extra "\n" before Thought. 
        # The previous loop adds a newline, so we append directly.
        thought_action = llm(prompt + f"Thought {i}:", stop=stop_sequences)

        try:
            # 3. ROBUST REGEX (Improved):
            # This regex allows for "Action 1:", "Action1:", or even just "Action:" 
            # (making the number optional via (\d+)?).
            pattern = re.compile(f"\\s*Action\\s*(\d+)?\\s*:\\s*", re.IGNORECASE)
            
            # split() will return the text before the pattern (Thought) 
            # and the text after (Action content).
            parts = pattern.split(thought_action.strip())
            
            # We expect [Thought, (optional number), Action string]
            # Because of the capturing group (\d+)?, split might return 3 elements.
            # We filter out None/Empty parts to get just Thought and Action.
            parts = [p for p in parts if p and p.strip() != str(i)]
            
            if len(parts) >= 2:
                thought = parts[0]
                action = parts[-1] # The last part is the action content
            else:
                raise ValueError("Format error")

        except Exception as e:
            # DEBUG PRINT: Uncomment this to see exactly how it's failing
            # print(f"DEBUG: Failed parse on: '{thought_action}' -> Error: {e}")
            
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            # Fallback: Force the model to generate just the action line
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()
        
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        
        if to_print:
            print(step_str)
        if done:
            break
            
    if not done:
        obs, r, done, info = step(env, "finish[]")
    
    if to_print:
        print(info, '\n')
        
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

def main():
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    num_sample = 50
    for i in idxs[:num_sample]:
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        infos.append(info)
        accuracy = sum(rs) / len(rs)
        avg_time = (time.time() - old_time) / len(rs)
        print(f"=" * 60)
        print(f"Progress: {len(rs)}/{num_sample} questions completed")
        print(f"Correct Answers: {sum(rs)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Time per Question: {avg_time:.2f} seconds")
        print(f"=" * 60)

if __name__ == "__main__":
    main()
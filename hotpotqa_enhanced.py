# ==========================================
# 1. SETUP & IMPORTS
# ==========================================
import os
import re
import openai
import random
import time
import json
import sys
import wikipedia
# Restore these imports to load the dataset!
import wikienv, wrappers 

openai.api_key = os.environ["OPENAI_API_KEY"]

# ==========================================
# 2. DEFINE THE TWO ENVIRONMENTS
# ==========================================

# --- A. The Data Loader (OLD) ---
# We use this ONLY to get questions (reset) and check answers (finish)
dataloader_env = wikienv.WikiEnv()
dataloader_env = wrappers.HotPotQAWrapper(dataloader_env, split="dev")
dataloader_env = wrappers.LoggingWrapper(dataloader_env)

# --- B. The Robust Searcher (NEW) ---
# We use this to actually search Wikipedia
import string

class RobustWikiEnv:
    def __init__(self):
        wikipedia.set_user_agent("ReAct-Replication-Agent/1.0 (academic purpose)")
        self.current_page_content = "" # STATE MEMORY

    def _extract_proper_nouns(self, text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        proper_nouns = [w for w in words if w[0].isupper() and w.lower() not in ['the', 'a', 'an', 'in', 'of', 'to']]
        return " ".join(proper_nouns)

    def step(self, action):
        action = action.strip()
        
        # =========================
        # HANDLE SEARCH
        # =========================
        if action.lower().startswith("search["):
            entity = action[7:-1].strip()
            
            # Logic: Try specific -> Try proper nouns -> Fail
            search_results = wikipedia.search(entity)
            if not search_results and len(entity.split()) > 3:
                simplified = self._extract_proper_nouns(entity)
                if simplified:
                    search_results = wikipedia.search(simplified)
            
            if not search_results:
                return f"Could not find [{entity}].", 0, False, {}

            try:
                # Load the page and SAVE CONTENT to state
                page = wikipedia.page(search_results[0], auto_suggest=False)
                self.current_page_content = page.content # <--- CRITICAL: Save full text
                
                # Return only the summary to start
                return page.summary[:500] + "\n(...use Lookup to read more)", 0, False, {}
            
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Ambiguous [{entity}]. Options: {e.options[:5]}", 0, False, {}
            except Exception as e:
                return f"Search error: {str(e)}", 0, False, {}

        # =========================
        # HANDLE LOOKUP (REAL IMPLEMENTATION)
        # =========================
        elif action.lower().startswith("lookup["):
            keyword = action[7:-1].strip()
            
            if not self.current_page_content:
                return "Error: No page loaded. Use Search first.", 0, False, {}
            
            # Simple Case-Insensitive 'Ctrl+F'
            # Find the sentence containing the keyword
            content_lower = self.current_page_content.lower()
            keyword_lower = keyword.lower()
            
            try:
                start_index = content_lower.index(keyword_lower)
                # Find the start of the sentence (last period before match)
                sent_start = self.current_page_content.rfind('.', 0, start_index) + 1
                # Find the end of the sentence (next period after match)
                sent_end = self.current_page_content.find('.', start_index) + 1
                
                if sent_end == 0: sent_end = len(self.current_page_content)
                
                # Return the specific sentence
                return f"Found: \"{self.current_page_content[sent_start:sent_end].strip()}\"", 0, False, {}
            except ValueError:
                return f"The term '{keyword}' was not found in the current page.", 0, False, {}

        elif action.lower().startswith("finish["):
            answer = action[7:-1]
            return answer, 1, True, {}
            
        return "Invalid Action.", 0, False, {}



action_env = RobustWikiEnv()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def llm(prompt, stop=["\n"], max_retries=10):
    for attempt in range(max_retries):
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
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

def clean_search_query(action):
    if not action.startswith("Search["):
        return action
    content = action[7:-1].strip('"').strip("'")
    stopwords = ["who", "is", "the", "of", "in", "what", "where", "search", "for"]
    words = content.split()
    if len(words) > 3 and words[0].lower() in ["who", "what", "where"]:
        keywords = [w for w in words if w[0].isupper()]
        if keywords:
            content = " ".join(keywords)
    return f"Search[{content}]"

def is_search_failure(obs):
    """
    Detects if the observation indicates a failed or ambiguous search.
    """
    failure_keywords = [
        "Could not find", 
        "Ambiguous", 
        "Search error", 
        "Invalid Action", 
        "similar to"
    ]
    return any(k in obs for k in failure_keywords)

# ==========================================
# 4. WEBTHINK (THE MAIN LOOP)
# ==========================================


folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.

CRITICAL RULES:
1. SEARCH ACTIONS: Use ONLY the specific entity name. Never write a sentence.
   - BAD: Search[American crime thriller film directed by Stuart Bird]
   - GOOD: Search[Stuart Bird]
   - BAD: Search[Who is the director of Titanic?]
   - GOOD: Search[Titanic (1997 film)]

2. AGE LOGIC: Remember that a SMALLER birth year means the person is OLDER.
   - 1960 < 1980, so the person born in 1960 is OLDER.

3. FINISH ACTIONS: Keep the answer brief.
   - If asked "Who created X?", provide the main creator only.
   - If asked "Did X come out first?", answer only with the name of the one that came first.

Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

# ==========================================
# REFACTORED WEBTHINK WITH REFLEXION
# ==========================================

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = dataloader_env.reset(idx=idx)
    done = False
    if to_print:
        print(idx, question)
    
    prompt += question + "\n"
    
    # 1. NEW: Track search history to prevent loops
    seen_searches = set() 
    
    n_calls, n_badcalls = 0, 0
    for i in range(1, 10):
        n_calls += 1
        
        # --- Generate Thought/Action ---
        stop_sequences = [f"\nObservation {i}:", f"\nObservation {i}", f"\nObservation:"]
        thought_action = llm(prompt + f"Thought {i}:", stop=stop_sequences)
        
        try:
            pattern = re.compile(f"\\s*Action\\s*(\d+)?\\s*:\\s*", re.IGNORECASE)
            parts = pattern.split(thought_action.strip())
            parts = [p for p in parts if p and p.strip() != str(i)]
            thought = parts[0]
            action = parts[-1]
        except:
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()

        action = clean_search_query(action) # Keep your existing cleaner

        # ============================================================
        # 2. CRITICAL FIX: SMART VALIDATOR
        # ============================================================
        validation_error = None
        
        # Check A: Infinite Loop Prevention
        if action.lower() in seen_searches:
            validation_error = f"Error: You already searched [{action}]. You must try a DIFFERENT search term."
        
        # Check B: Length/Format Prevention (stops the "sentence search" bug)
        elif action.lower().startswith("search[") and len(action.split()) > 10:
            validation_error = f"Error: Search query is too long. Use KEYWORDS only (e.g., Search[Name])."

        if validation_error:
            # Skip the actual API call, feed the error back to the agent immediately
            obs = validation_error
            # Don't add to seen_searches if it was a validator error
        else:
            # If valid, record it and execute
            seen_searches.add(action.lower())
            
            if action.lower().startswith("finish"):
                try:
                    ans = action[action.find("[")+1 : action.rfind("]")]
                    obs, r, done, info = dataloader_env.step(f"finish[{ans}]")
                except:
                    obs, r, done, info = dataloader_env.step(action.lower())
            else:
                obs, r, done, info = action_env.step(action)

        # ============================================================

        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        
        if to_print:
            print(step_str)
        
        if done:
            break
            
    if not done:
        obs, r, done, info = dataloader_env.step("finish[]")
    
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info


# ==========================================
# 5. MAIN
# ==========================================
def main():
    # Ensure this range matches your dataset size
    idxs = list(range(7405)) 
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    num_sample = 500
    
    for i in idxs[:num_sample]:
        try:
            r, info = webthink(i, to_print=True)
            rs.append(info['em'])
            infos.append(info)
        except Exception as e:
            print(f"Skipping index {i} due to error: {e}")
            continue

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
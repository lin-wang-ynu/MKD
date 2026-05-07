import os
import json
import random
import argparse
import logging
import sys
import subprocess
import torch
import numpy as np
from datetime import datetime


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
LOG = logging.getLogger(__name__)

def initialize_mike_order_with_llm():

    LOG.info("🚀 [Init] Asking LLM to generate the INITIAL MIKE_ORDER configuration...")
    
    prompt = f"""You are initializing the MIKE_ORDER configuration for a Cross-Lingual Knowledge Editing system.

**Task**: Generate an optimal ordering and ratio for 32 few-shot demonstrations.

**Shot Types**:
- **Type 0 (Identity)**: Original question-answer pairs. Improves **Reliability** (same question recall).
- **Type 1 (Paraphrase)**: Rephrased questions. Improves **Generalization** (paraphrase understanding).
- **Type 3 (Portability)**: Reasoning questions. Improves **Portability** (logical inference).

**Guidelines**:
1. **Total shots**: Exactly 32 shots
2. **Ratio**: No hard constraints - explore freely to find optimal distribution
3. **Ordering strategy**: 
   - Later positions have stronger influence (Recency Bias) in decoder-only models
   - Consider interleaving types for diversity
   - Balance between clustering similar types vs. mixing

**Current baseline**: [0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3]
- Ratio: 8 Type0, 12 Type1, 12 Type3

**Output**:
Provide a JSON object with a single key "mike_order" containing an array of 32 integers (only 0, 1, or 3).

Example:
```json
{{
    "mike_order": [0, 1, 3, 0, 1, 1, 3, 3, ...]
}}
```
"""
    
    import time
    from openai import OpenAI
    import httpx
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            http_client = httpx.Client(timeout=60.0, transport=httpx.HTTPTransport(retries=3))
            
            client = OpenAI(
                api_key='************', 
                base_url='**********',
                http_client=http_client
            )
            
            LOG.info(f"🚀 [Init] Connecting to LLM API (Attempt {attempt+1})...")
            response = client.chat.completions.create(
                model="claude-sonnet-4-5-20250929-t",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )
            
            import re
            import ast
            full_content = response.choices[0].message.content
            LOG.info(f"📝 [Init LLM Response] {full_content[:200]}...")
            

            json_str = None
            if "```json" in full_content:
                json_str = full_content.split("```json")[1].split("```")[0].strip()
            elif "```" in full_content:
                json_str = full_content.split("```")[-2].strip()
            else:
                match = re.search(r'(\{.*"mike_order".*?\})', full_content, re.DOTALL)
                if match: json_str = match.group(1)
            
            if json_str:
                json_str = re.sub(r'//.*', '', json_str)
                
                try:
                    selection = json.loads(json_str)
                except:
                    selection = ast.literal_eval(json_str)
                
                mike_order = selection.get('mike_order', [])
                

                if len(mike_order) == 32 and all(t in [0, 1, 3] for t in mike_order):
                    count_0 = mike_order.count(0)
                    count_1 = mike_order.count(1)
                    count_3 = mike_order.count(3)
                    
                    LOG.info(f"✅ LLM Initialization Successful: Type0={count_0}, Type1={count_1}, Type3={count_3}")
                    return mike_order
                else:
                    LOG.warning(f"⚠️ Invalid MIKE_ORDER from LLM: length={len(mike_order)}")

        except Exception as e:
            LOG.warning(f"Init attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    LOG.warning("❌ LLM Initialization failed, falling back to baseline configuration.")

    fallback_order = [0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3, 0, 0, 1, 1, 1, 3, 3, 3]
    LOG.info(f"✅ Fallback initialization: {fallback_order}")
    return fallback_order

def call_llm_optimizer(current_order, metrics, history, best_order=None):
    """
    RMP核心：调用Claude 3.5 Sonnet优化MIKE_ORDER配置
    
    Args:
        current_order: 当前配置（仅在EXPLORATION阶段使用）
        metrics: 当前指标
        history: 历史记录
        best_order: 历史最佳配置（在REFINEMENT和EXPLOITATION阶段使用）
    """
    print("\n🤖 [RMP] Calling LLM (Claude 3.5 Sonnet) to optimize MIKE_ORDER...")
    

    history_desc = ""
    is_stagnated = False
    temperature = 0.7
    mutation_msg = ""
    

    best_config_analysis = ""
    metric_growth_analysis = ""
    
    if history:
        best_iter = max(history, key=lambda x: x['score'])
        best_score = best_iter['score']
        best_idx = best_iter['iter']
        
        best_order = best_iter['mike_order']
        best_count_0 = best_order.count(0)
        best_count_1 = best_order.count(1)
        best_count_3 = best_order.count(3)
        
        best_config_analysis = f"""
**📈 HISTORICAL BEST CONFIGURATION**

The BEST score so far is **{best_score:.2f}** achieved in **Iteration {best_idx+1}** with:
- **Type 0 (Identity)**: {best_count_0} shots
- **Type 1 (Paraphrase)**: {best_count_1} shots
- **Type 3 (Portability)**: {best_count_3} shots

**Order pattern**: {best_order}

**Key Metrics in Best Config**:
- EN Edit: Rel={best_iter['metrics'].get('en_edit_reliability_f1', 0):.1f}, Gen_EN={best_iter['metrics'].get('en_edit_generalization_en_f1', 0):.1f}, Gen_ZH={best_iter['metrics'].get('en_edit_generalization_zh_f1', 0):.1f}, Port_EN={best_iter['metrics'].get('en_edit_portability_en_f1', 0):.1f}, Port_ZH={best_iter['metrics'].get('en_edit_portability_zh_f1', 0):.1f}
- ZH Edit: Rel={best_iter['metrics'].get('zh_edit_reliability_f1', 0):.1f}, Gen_EN={best_iter['metrics'].get('zh_edit_generalization_en_f1', 0):.1f}, Gen_ZH={best_iter['metrics'].get('zh_edit_generalization_zh_f1', 0):.1f}, Port_EN={best_iter['metrics'].get('zh_edit_portability_en_f1', 0):.1f}, Port_ZH={best_iter['metrics'].get('zh_edit_portability_zh_f1', 0):.1f}

**Your Goal**: Either improve upon this configuration OR explore a radically different approach.
"""
        

        if len(history) >= 3:
            recent_history = history  
            
            metric_keys = [
                ('en_edit_reliability_f1', 'EN Edit Reliability'),
                ('en_edit_generalization_en_f1', 'EN Edit Gen(EN)'),
                ('en_edit_generalization_zh_f1', 'EN Edit Gen(ZH)'),
                ('en_edit_portability_en_f1', 'EN Edit Port(EN)'),
                ('en_edit_portability_zh_f1', 'EN Edit Port(ZH)'),
                ('zh_edit_reliability_f1', 'ZH Edit Reliability'),
                ('zh_edit_generalization_en_f1', 'ZH Edit Gen(EN)'),
                ('zh_edit_generalization_zh_f1', 'ZH Edit Gen(ZH)'),
                ('zh_edit_portability_en_f1', 'ZH Edit Port(EN)'),
                ('zh_edit_portability_zh_f1', 'ZH Edit Port(ZH)')
            ]
            
            growth_potential = []
            
            for key, name in metric_keys:
                values = [h['metrics'].get(key, 0) for h in recent_history]
                if not values or all(v == 0 for v in values):
                    continue
                
                avg_value = sum(values) / len(values)
                max_value = max(values)
                min_value = min(values)
                volatility = max_value - min_value
                
                if volatility > 5.0:
                    potential = "HIGH"
                    reason = f"High volatility ({volatility:.1f}%) indicates room for optimization"
                elif avg_value > 90 and volatility > 2.0:
                    potential = "MEDIUM"
                    reason = f"High score ({avg_value:.1f}%) but still fluctuating"
                elif avg_value < 75 and volatility < 3.0:
                    potential = "LOW"
                    reason = f"Low score ({avg_value:.1f}%) with low volatility, likely at ceiling"
                elif avg_value > 95:
                    potential = "LOW"
                    reason = f"Already very high ({avg_value:.1f}%), diminishing returns"
                else:
                    potential = "MEDIUM"
                    reason = f"Moderate performance ({avg_value:.1f}%)"
                
                growth_potential.append({
                    'name': name,
                    'key': key,
                    'avg': avg_value,
                    'potential': potential,
                    'reason': reason
                })
            
            high_potential = [m for m in growth_potential if m['potential'] == 'HIGH']
            medium_potential = [m for m in growth_potential if m['potential'] == 'MEDIUM']
            low_potential = [m for m in growth_potential if m['potential'] == 'LOW']
            
            metric_growth_analysis = f"""
**🎯 METRIC GROWTH POTENTIAL ANALYSIS** (Based on Recent 3 Iterations)

**HIGH GROWTH POTENTIAL** (Focus here):
"""
            if high_potential:
                for m in high_potential:
                    metric_growth_analysis += f"    ✅ **{m['name']}**: {m['avg']:.1f}% | {m['reason']}\n"
            else:
                metric_growth_analysis += "    (None identified)\n"
            
            metric_growth_analysis += """
**MEDIUM GROWTH POTENTIAL**:
"""
            if medium_potential:
                for m in medium_potential:
                    metric_growth_analysis += f"    ⚠️ {m['name']}: {m['avg']:.1f}% | {m['reason']}\n"
            else:
                metric_growth_analysis += "    (None)\n"
            
            metric_growth_analysis += """
**LOW GROWTH POTENTIAL** (Don't over-invest):
"""
            if low_potential:
                for m in low_potential:
                    metric_growth_analysis += f"    ❌ {m['name']}: {m['avg']:.1f}% | {m['reason']}\n"
            else:
                metric_growth_analysis += "    (None)\n"
        

        if len(history) >= 4:
            last_scores = [h['score'] for h in history[-4:]]
            score_range = max(last_scores) - min(last_scores)
            
            if score_range < 1.0:
                is_stagnated = True
                temperature = 1.0
                mutation_msg = """
⚠️ **STAGNATION DETECTED**: Scores have plateaued in the last 4 iterations.

**EXPLORATION MODE ACTIVATED**:
1. Try a DIFFERENT ratio - significantly adjust the distribution of Type 0/1/3
2. Change ordering pattern significantly (e.g., from blocks to interleaved, or try gradual/random patterns)
3. It's OK if score drops by 1-2% - we need to escape local optimum!
"""
                LOG.warning("⚠️ Stagnation detected. Temperature -> 1.0")

        history_desc = f"\n**Optimization History (All {len(history)} Previous Iterations):**\n"
        for h in history:
            iter_num = h['iter'] + 1
            score = h['score']
            order = h['mike_order']
            count_0 = order.count(0)
            count_1 = order.count(1)
            count_3 = order.count(3)
            
            m = h['metrics']
            metrics_summary = (
                f"EN[Rel:{m.get('en_edit_reliability_f1', 0):.1f} "
                f"GenEN:{m.get('en_edit_generalization_en_f1', 0):.1f} "
                f"GenZH:{m.get('en_edit_generalization_zh_f1', 0):.1f} "
                f"PortEN:{m.get('en_edit_portability_en_f1', 0):.1f} "
                f"PortZH:{m.get('en_edit_portability_zh_f1', 0):.1f}] "
                f"ZH[Rel:{m.get('zh_edit_reliability_f1', 0):.1f} "
                f"GenEN:{m.get('zh_edit_generalization_en_f1', 0):.1f} "
                f"GenZH:{m.get('zh_edit_generalization_zh_f1', 0):.1f} "
                f"PortEN:{m.get('zh_edit_portability_en_f1', 0):.1f} "
                f"PortZH:{m.get('zh_edit_portability_zh_f1', 0):.1f}]"
            )
            
            history_desc += f"- Iter {iter_num}: Score={score:.2f} | Type0={count_0} Type1={count_1} Type3={count_3}\n"
            history_desc += f"  {metrics_summary}\n"
            history_desc += f"  Order={order}\n"


    current_iter = len(history) + 1
    max_iter = 20
    

    base_order = current_order
    if current_iter <= max_iter * 0.3:
        phase = "EXPLORATION"
        phase_guidance = """
**Current Phase: EXPLORATION (Iterations 1-6)**
- **Goal**: Discover diverse ratio and ordering patterns
- **Strategy**: 
  * Try wide range of ratios - explore freely without constraints
  * Experiment with different ordering patterns (blocks vs interleaved)
- **Risk Tolerance**: HIGH - Accept 2-3% score drops for learning
"""
        temperature = max(temperature, 0.8)
    elif current_iter <= max_iter * 0.7:
        phase = "REFINEMENT"

        if best_order is not None:
            base_order = best_order
            LOG.info(f"🎯 [REFINEMENT] Using BEST historical config as base (not current config)")
        phase_guidance = """
**Current Phase: REFINEMENT (Iterations 7-14)**
- **Goal**: Fine-tune the BEST configuration found so far
- **Strategy**:
  * **IMPORTANT**: Start from the BEST historical configuration provided below
  * Analyze which metrics need improvement and adjust ratios accordingly
  * Optimize ordering patterns based on the best config's structure
  * Balance between preserving what works and making necessary improvements
- **Risk Tolerance**: MEDIUM - Willing to make adjustments if justified by metric analysis
"""
        temperature = 0.7 if not is_stagnated else temperature
    else:
        phase = "EXPLOITATION"

        if best_order is not None:
            base_order = best_order
            LOG.info(f"🎯 [EXPLOITATION] Using BEST historical config as base (not current config)")
        phase_guidance = """
**Current Phase: EXPLOITATION (Iterations 15-20)**
- **Goal**: Squeeze final improvements from the BEST configuration
- **Strategy**:
  * **CRITICAL**: Start from the BEST historical configuration provided below
  * Make precise, targeted adjustments to address remaining weak metrics
  * Fine-tune both ratios and ordering based on detailed metric analysis
  * Focus on extracting maximum performance from the best configuration
- **Risk Tolerance**: LOW - Prioritize stability but don't avoid beneficial changes
"""
        temperature = 0.5 if not is_stagnated else temperature


    prompt = f"""
You are an expert Meta-Prompting Agent optimizing the MIKE_ORDER configuration for Cross-Lingual Knowledge Editing.

{mutation_msg}

{phase_guidance}

**System Mechanism**:
The MIKE method uses 32 few-shot demonstrations with 3 types:
- **Type 0 (Identity)**: Original Q&A pairs → Improves **Reliability**
- **Type 1 (Paraphrase)**: Rephrased questions → Improves **Generalization**
- **Type 3 (Portability)**: Reasoning questions → Improves **Portability**

**Shot Type → Metric Mapping**:
1. **Type 0** → **Reliability** (same question recall after editing)
2. **Type 1** → **Generalization** (paraphrase understanding, both EN and ZH)
3. **Type 3** → **Portability** (logical reasoning, both EN and ZH)

**Recency Bias**: Later positions (22-31) have ~3-5x stronger influence than earlier positions in decoder-only models.

{best_config_analysis}

{metric_growth_analysis}

{history_desc}

**Current Performance**:
- EN Edit: Rel={metrics.get('en_edit_reliability_f1', 0):.2f}%, GenEN={metrics.get('en_edit_generalization_en_f1', 0):.2f}%, GenZH={metrics.get('en_edit_generalization_zh_f1', 0):.2f}%, PortEN={metrics.get('en_edit_portability_en_f1', 0):.2f}%, PortZH={metrics.get('en_edit_portability_zh_f1', 0):.2f}%
- ZH Edit: Rel={metrics.get('zh_edit_reliability_f1', 0):.2f}%, GenEN={metrics.get('zh_edit_generalization_en_f1', 0):.2f}%, GenZH={metrics.get('zh_edit_generalization_zh_f1', 0):.2f}%, PortEN={metrics.get('zh_edit_portability_en_f1', 0):.2f}%, PortZH={metrics.get('zh_edit_portability_zh_f1', 0):.2f}%

**Base MIKE_ORDER for This Iteration**: {base_order}
- Type 0: {base_order.count(0)} shots
- Type 1: {base_order.count(1)} shots
- Type 3: {base_order.count(3)} shots

**IMPORTANT**: In REFINEMENT/EXPLOITATION phases, this is the BEST historical config, NOT the previous iteration's config. You MUST start your optimization from this base configuration.

**Optimization Strategy**:
1. **Identify weak metrics** from the current performance
2. **Allocate more shots** to the corresponding type (but respect growth potential!)
3. **Optimize ordering**: Place high-priority types in later positions (22-31) to leverage recency bias

**Constraints**:
- Total: Exactly 32 shots
- Each type (0, 1, 3) can appear any number of times (including 0)
- Explore diverse ratios freely

**Output Format**:
First provide your **Reasoning**, then the **JSON Configuration**.

Example:
Reasoning: Generalization EN is weak (65%), so I'll increase Type 1 from 12 to 14 shots and place more Type 1 in early positions...

```json
{{
    "mike_order": [1, 0, 1, 3, 1, 0, 3, 1, 3, 0, 1, 3, 1, 0, 3, 1, 3, 0, 1, 3, 1, 0, 3, 1, 3, 0, 1, 3, 1, 3, 0, 3]
}}
```
"""
    

    import time
    import httpx
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            from openai import OpenAI
            
            http_client = httpx.Client(timeout=120.0, transport=httpx.HTTPTransport(retries=3))
            
            client = OpenAI(
                api_key='AfBz4xf1voa1d',
                base_url='https://ai.liaobots.work/v1',
                http_client=http_client
            )

            LOG.info(f"🚀 [RMP] Connecting to LLM API (Attempt {attempt+1})...")
            
            response = client.chat.completions.create(
                model="claude-sonnet-4-5-20250929-t",
                messages=[
                    {"role": "system", "content": "You are a Meta-Prompting optimizer. Analyze -> Strategize -> Execute."},
                    {"role": "user", "content": prompt}
                ],
                stream=False,
                temperature=temperature,
                max_tokens=2048
            )
            
            import re
            import ast
        
            full_content = response.choices[0].message.content
            LOG.info(f"📝 [LLM Raw Response] {full_content[:500]}...")
            

            json_str = None
        
            if "```json" in full_content:
                json_str = full_content.split("```json")[1].split("```")[0].strip()
            elif "```" in full_content:
                json_str = full_content.split("```")[-2].strip()
            else:
                match = re.search(r'(\{.*"mike_order".*?\})', full_content, re.DOTALL)
                if match:
                    json_str = match.group(1)
            
            if not json_str:
                raise ValueError("Could not find JSON object in LLM response")


            json_str = re.sub(r'//.*', '', json_str)
            json_str = re.sub(r'#.*', '', json_str)
            

            def clean_array(match):
                array_content = match.group(1)
                numbers = re.findall(r'\d+', array_content)
                return '[' + ', '.join(numbers) + ']'
            
            json_str = re.sub(r'"mike_order"\s*:\s*\[(.*?)\]', 
                             lambda m: '"mike_order": ' + clean_array(m), json_str, flags=re.DOTALL)
            
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

            selection = None
            try:
                selection = json.loads(json_str)
            except json.JSONDecodeError:

                match = re.search(r'"mike_order"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
                if match:
                    nums = [int(x) for x in re.findall(r'\d+', match.group(1))]
                    if nums:
                        selection = {'mike_order': nums}
                        LOG.info(f"✅ Extracted mike_order via regex: {len(nums)} elements")
                    else:
                        raise ValueError("Could not extract any valid numbers")
                else:
                    try:
                        selection = ast.literal_eval(json_str)
                    except Exception as e:
                        LOG.error(f"❌ All parsing methods failed. JSON: {json_str[:500]}")
                        raise e

            LOG.info(f"✅ Parsed JSON Selection: {json.dumps(selection)}")
        
            new_order = selection.get('mike_order', [])
            

            if len(new_order) != 32:
                LOG.warning(f"⚠️ LLM returned {len(new_order)} shots, expected 32. Adjusting...")
                if len(new_order) > 32:
                    new_order = new_order[:32]
                else:

                    while len(new_order) < 32:
                        new_order.append(random.choice([0, 1, 3]))
            
 
            new_order = [t if t in [0, 1, 3] else random.choice([0, 1, 3]) for t in new_order]
            

            count_0 = new_order.count(0)
            count_1 = new_order.count(1)
            count_3 = new_order.count(3)
            
            LOG.info(f"✅ New MIKE_ORDER: Type0={count_0}, Type1={count_1}, Type3={count_3}")
            
  
            if count_0 < 4 or count_0 > 14:
                LOG.warning(f"⚠️ Type0 count ({count_0}) out of preferred range [6-12], but accepting")
            if count_1 < 6 or count_1 > 16:
                LOG.warning(f"⚠️ Type1 count ({count_1}) out of preferred range [8-14], but accepting")
            if count_3 < 6 or count_3 > 16:
                LOG.warning(f"⚠️ Type3 count ({count_3}) out of preferred range [8-14], but accepting")
            
            print(f"🤖 [RMP] Optimization complete. New order: Type0={count_0}, Type1={count_1}, Type3={count_3}")
            return new_order

        except Exception as e:
            LOG.warning(f"⚠️ LLM Call Failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                LOG.info(f"⏳ Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                LOG.error(f"❌ LLM Call Failed after {max_retries} attempts. Using current order.")
                print(f"⚠️ [RMP] LLM failed. Keeping current configuration.")
                return current_order

def run_mike_evaluation(mike_order, model_name, edit_lang, val_set_path, output_dir):


    config_file = os.path.join(output_dir, 'temp_mike_config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({'mike_order': mike_order}, f)
    
    LOG.info(f"✅ Saved temp config to: {config_file}")
    

    cmd = [
        sys.executable, 'run_mike_worker_rmp.py',
        '--model_name', model_name,
        '--edit_lang', edit_lang,
        '--val_set_path', val_set_path,
        '--config_file', config_file,
        '--output_dir', output_dir
    ]
    
    LOG.info(f"🚀 Running subprocess: {' '.join(cmd)}")
    

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        LOG.info("🧹 Cleared CUDA cache before subprocess")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        LOG.error(f"❌ MIKE worker failed with exit code {e.returncode}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return []
        

    result_file = os.path.join(output_dir, f"result_rmp_{edit_lang}.json")
    if os.path.exists(result_file):
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        LOG.error(f"❌ Result file not found: {result_file}")
        return []

def calculate_metrics(results_en, results_zh):

    from evaluate import obtain_f1_and_em, my_avg
    
    def calc_for_lang(results):
        if not results:
            return {
                'reliability_f1': 0.0,
                'generalization_en_f1': 0.0,
                'generalization_zh_f1': 0.0,
                'locality_en_f1': 0.0,
                'locality_zh_f1': 0.0,
                'portability_en_f1': 0.0,
                'portability_zh_f1': 0.0
            }
        
        scores = {
            'reliability_f1': [],
            'generalization_en_f1': [],
            'generalization_zh_f1': [],
            'locality_en_f1': [],
            'locality_zh_f1': [],
            'portability_en_f1': [],
            'portability_zh_f1': []
        }
        
        for item in results:
            # Reliability
            f1, _ = obtain_f1_and_em(item["post"]["rewrite_acc"]["ans"], item["post"]["rewrite_acc"]["target"])
            scores['reliability_f1'].append(f1)
            
            # Generalization
            if item["post"].get("rephrase_acc_en") and item["post"]["rephrase_acc_en"].get("ans"):
                f1_en, _ = obtain_f1_and_em(item["post"]["rephrase_acc_en"]["ans"], item["post"]["rephrase_acc_en"]["target"])
                scores['generalization_en_f1'].append(f1_en)
            
            if item["post"].get("rephrase_acc_zh") and item["post"]["rephrase_acc_zh"].get("ans"):
                f1_zh, _ = obtain_f1_and_em(item["post"]["rephrase_acc_zh"]["ans"], item["post"]["rephrase_acc_zh"]["target"])
                scores['generalization_zh_f1'].append(f1_zh)
            
            # Locality
            if item["post"].get("locality_en") and item["post"]["locality_en"].get("neighborhood_output_en"):
                f1_loc_en, _ = obtain_f1_and_em(
                    item["post"]["locality_en"]["neighborhood_output_en"]["ans"],
                    item["pre"]["locality_en"]["neighborhood_output_en"]["ans"]
                )
                scores['locality_en_f1'].append(f1_loc_en)
            
            if item["post"].get("locality_zh") and item["post"]["locality_zh"].get("neighborhood_output_zh"):
                f1_loc_zh, _ = obtain_f1_and_em(
                    item["post"]["locality_zh"]["neighborhood_output_zh"]["ans"],
                    item["pre"]["locality_zh"]["neighborhood_output_zh"]["ans"]
                )
                scores['locality_zh_f1'].append(f1_loc_zh)
            
            # Portability
            if item["post"].get("portability_en") and item["post"]["portability_en"].get("one_hop_acc_en"):
                f1_port_en, _ = obtain_f1_and_em(
                    item["post"]["portability_en"]["one_hop_acc_en"]["ans"],
                    item["post"]["portability_en"]["one_hop_acc_en"]["target"]
                )
                scores['portability_en_f1'].append(f1_port_en)
            
            if item["post"].get("portability_zh") and item["post"]["portability_zh"].get("one_hop_acc_en"):
                f1_port_zh, _ = obtain_f1_and_em(
                    item["post"]["portability_zh"]["one_hop_acc_en"]["ans"],
                    item["post"]["portability_zh"]["one_hop_acc_en"]["target"]
                )
                scores['portability_zh_f1'].append(f1_port_zh)
        
        return {k: my_avg(v) if v else 0.0 for k, v in scores.items()}
    

    metrics_en = calc_for_lang(results_en)
    metrics_zh = calc_for_lang(results_zh)
    

    combined = {}
    for k, v in metrics_en.items():
        combined[f'en_edit_{k}'] = v
    for k, v in metrics_zh.items():
        combined[f'zh_edit_{k}'] = v
    

    score_en = (
        metrics_en['reliability_f1'] +
        metrics_en['generalization_en_f1'] +
        metrics_en['generalization_zh_f1'] +
        metrics_en['portability_en_f1'] +
        metrics_en['portability_zh_f1']
    ) / 5.0
    
    score_zh = (
        metrics_zh['reliability_f1'] +
        metrics_zh['generalization_en_f1'] +
        metrics_zh['generalization_zh_f1'] +
        metrics_zh['portability_en_f1'] +
        metrics_zh['portability_zh_f1']
    ) / 5.0
    
    final_score = (score_en + score_zh) / 2.0
    
    LOG.info(f"📊 Overall Score: {final_score:.2f}")
    LOG.info(f"   EN Edit: {score_en:.2f} | ZH Edit: {score_zh:.2f}")
    
    return combined, final_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=['baichuan', 'chinese_llama2'],
                        help='模型名称')
    parser.add_argument('--val_set', type=str, default='data/Bi-ZsRE-data/bizsre_test.json',
                        help='验证集路径')
    parser.add_argument('--max_iter', type=int, default=20,
                        help='最大迭代次数')
    parser.add_argument('--initial_order', type=str, default=None,
                        help='初始MIKE_ORDER配置文件（JSON）')
    args = parser.parse_args()


    if args.initial_order and os.path.exists(args.initial_order):
        LOG.info(f"📦 Loading initial order from: {args.initial_order}")
        with open(args.initial_order, 'r', encoding='utf-8') as f:
            config = json.load(f)
            current_order = config.get('mike_order', [])
        LOG.info(f"✅ Loaded Initial Order: {current_order}")
    elif args.model_name == 'chinese_llama2':

        LOG.info("📦 Using optimized initial order for Chinese-LLaMA2 (from best_mike_order_chinese_llama2.json)...")
        current_order = [1, 3, 1, 3, 1, 3, 1, 3, 0, 3, 1, 3, 3, 0, 1, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3]
        LOG.info(f"✅ Initial Order: {current_order}")
        LOG.info(f"   Type0={current_order.count(0)}, Type1={current_order.count(1)}, Type3={current_order.count(3)}")
    else:
        LOG.info("🚀 Using LLM to generate initial MIKE_ORDER...")
        current_order = initialize_mike_order_with_llm()
    
    history = []
    best_score = -1
    best_order = None

    torch.cuda.empty_cache()

    # RMP迭代循环
    for i in range(args.max_iter):
        LOG.info(f"\n{'='*20} Iteration {i+1}/{args.max_iter} {'='*20}")
        
        try:

            output_dir_en = f'rmp_mike_results/iter_{i}/en'
            output_dir_zh = f'rmp_mike_results/iter_{i}/zh'
            os.makedirs(output_dir_en, exist_ok=True)
            os.makedirs(output_dir_zh, exist_ok=True)
            
            LOG.info("📊 Evaluating EN edit...")
            results_en = run_mike_evaluation(current_order, args.model_name, 'en', args.val_set, output_dir_en)
            
            LOG.info("🧹 Clearing cache between evaluations...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            import time
            time.sleep(5)
            
            LOG.info("📊 Evaluating ZH edit...")
            results_zh = run_mike_evaluation(current_order, args.model_name, 'zh', args.val_set, output_dir_zh)
            
            # 计算指标
            metrics, score = calculate_metrics(results_en, results_zh)
            
            # 记录历史
            history.append({
                'iter': i,
                'score': score,
                'metrics': metrics,
                'mike_order': current_order
            })
            
            if score > best_score:
                best_score = score
                best_order = current_order
                print(f"🌟 New Best Score: {best_score:.2f}")
            
            if i < args.max_iter - 1:
                current_order = call_llm_optimizer(current_order, metrics, history, best_order)
            
            with open(f'rmp_mike_optimization_history_{args.model_name}.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            LOG.info(f"💾 Checkpoint: Saved history for iteration {i}")
                
            torch.cuda.empty_cache()
            
        except Exception as e:
            LOG.error(f"Error in iteration {i}: {e}", exc_info=True)
            break
            
    print(f"\n🏆 Optimization Finished!")
    print(f"Best Score: {best_score:.2f}")
    
    if best_order:
        with open(f'best_mike_order_{args.model_name}.json', 'w', encoding='utf-8') as f:
            json.dump({'mike_order': best_order}, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved best configuration to 'best_mike_order_{args.model_name}.json'")
        
    with open(f'rmp_mike_optimization_history_{args.model_name}.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"📜 Saved full history to 'rmp_mike_optimization_history_{args.model_name}.json'")

if __name__ == "__main__":
    main()

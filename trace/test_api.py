import json
import os
import argparse
import torch
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# ================= 配置区域 =================

# API 配置
API_KEY = ''
BASE_URL = ''
MODEL_NAME = ""


SIMILARITY_THRESHOLD = 0.7
CLASSIFIER_MODEL_PATH = 'model/xlm-roberta-base'
CLASSIFIER_ARCHIVE = 'models/multilingual_similarity_classifier.pth'

INPUT_FILE_PATH = "data/Bi-ZsRE-data/bizsre_test.json"
FEW_SHOT_FILE = "few_shot_examples.json"
RESULTS_DIR = "baseline/results"

# ================= 初始化 =================


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"正在加载分类器模型: {CLASSIFIER_MODEL_PATH}")
print(f"加载预训练权重: {CLASSIFIER_ARCHIVE}")
try:
    from transformers import XLMRobertaModel, XLMRobertaTokenizer
    
    classifier_tok = XLMRobertaTokenizer.from_pretrained(CLASSIFIER_MODEL_PATH)
    classifier_model = XLMRobertaModel.from_pretrained(CLASSIFIER_MODEL_PATH)
    

    if os.path.exists(CLASSIFIER_ARCHIVE):
        print(f"加载训练好的分类器权重...")
        checkpoint = torch.load(CLASSIFIER_ARCHIVE, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            classifier_model.load_state_dict(checkpoint['model_state_dict'])
            print("分类器权重加载成功（从 model_state_dict）")
        elif 'model' in checkpoint:
            classifier_model.load_state_dict(checkpoint['model'])
            print("分类器权重加载成功（从 model）")
        else:
            classifier_model.load_state_dict(checkpoint)
            print("分类器权重加载成功（直接加载）")
    else:
        print(f"警告: 未找到训练好的权重文件 {CLASSIFIER_ARCHIVE}，使用预训练权重")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier_model.to(device)
    classifier_model.eval()
    print(f"分类器模型加载完成，使用设备: {device}")
except Exception as e:
    print(f"加载分类器模型失败: {e}")
    import traceback
    traceback.print_exc()
    exit()

print(f"正在加载 few-shot examples: {FEW_SHOT_FILE}")
try:
    with open(FEW_SHOT_FILE, 'r', encoding='utf-8') as f:
        few_shot_data = json.load(f)
    print(f"Few-shot examples 加载完成: 中文 {len(few_shot_data['zh'])} 个，英文 {len(few_shot_data['en'])} 个")
except Exception as e:
    print(f"加载 few-shot examples 失败: {e}")
    exit()



def get_embedding(text_list):

    encoded_input = classifier_tok(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = classifier_model(**encoded_input)

    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    
    return mean_embeddings


def query_api(prompt, max_tokens=4096):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""


class Memory:
    
    def __init__(self):
        self.questions = []
        self.answers = []
        self.embeddings = None

    def add(self, question, answer):
        self.questions.append(question)
        self.answers.append(answer)
        self.embeddings = get_embedding(self.questions)
    
    def search(self, query):

        if len(self.questions) == 0:
            return 0.0, -1

        query_embedding = get_embedding([query])
        cache_embeddings = self.embeddings

        query_embedding = query_embedding.unsqueeze(1)
        cache_embeddings = cache_embeddings.unsqueeze(1)
        

        cos = (cache_embeddings[None] * query_embedding[:, None]).sum(-1) / \
              (cache_embeddings[None].norm(2, -1) * query_embedding[:, None].norm(2, -1))

        cos = torch.clamp(cos, -1.0, 1.0)

        normalized_sim = (cos + 1) / 2
        normalized_sim = torch.clamp(normalized_sim, 0.0, 1.0)

        normalized_sim = normalized_sim.squeeze()  # [num_cache]
        
        max_score = normalized_sim.max().item()
        max_idx = normalized_sim.argmax().item()
        
        return max_score, max_idx
    
    def clear(self):
        self.questions = []
        self.answers = []
        self.embeddings = None


def construct_trace_prompt(question, memory, few_shot_examples, lang='en'):

    similarity, idx = memory.search(question)
    if similarity < SIMILARITY_THRESHOLD:
        return question, similarity

    few_shot_prefix = "\n\n".join(few_shot_examples)

    recalled_question = memory.questions[idx]
    recalled_answer = memory.answers[idx]
    new_fact = f"New Fact: Question: {recalled_question} Answer: {recalled_answer}"

    current_prompt = f"Prompt: Question: {question} Answer:"
    full_prompt = f"{few_shot_prefix}\n\n{new_fact}\n{current_prompt}"
    
    return full_prompt, similarity


def test_single_question(question, memory, few_shot_examples, lang='en'):
    prompt, similarity = construct_trace_prompt(question, memory, few_shot_examples, lang)
    answer = query_api(prompt)
    return answer, similarity


def run_trace_evaluation(data, edit_lang='en', limit=200):

    test_data = data[:limit]
    results = []

    memory = Memory()

    few_shot_en = few_shot_data['en']
    few_shot_zh = few_shot_data['zh']
    
    print(f"\n开始 评估")
    print(f"编辑语言: {edit_lang}")
    print(f"测试数据: {len(test_data)} 条")
    print(f"相似度阈值: {SIMILARITY_THRESHOLD}")
    print("="*80)
    
    for idx, item in tqdm(enumerate(test_data), total=len(test_data), desc="Evaluation"):
        result_item = item.copy()

        pre_results = {}
        
        for test_lang in ['en', 'zh']:
            if test_lang not in item:
                continue
            
            lang_data = item[test_lang]
            few_shot_examples = few_shot_en if test_lang == 'en' else few_shot_zh

            if 'src' in lang_data and lang_data['src']:
                ans, sim = test_single_question(lang_data['src'], memory, few_shot_examples, test_lang)
                pre_results[f'src_{test_lang}'] = {
                    'question': lang_data['src'],
                    'target': lang_data.get('alt', ''),
                    'ans': ans,
                    'similarity': sim
                }

            if 'rephrase' in lang_data and lang_data['rephrase']:
                ans, sim = test_single_question(lang_data['rephrase'], memory, few_shot_examples, test_lang)
                pre_results[f'rephrase_{test_lang}'] = {
                    'question': lang_data['rephrase'],
                    'target': lang_data.get('alt', ''),
                    'ans': ans,
                    'similarity': sim
                }

            if 'loc' in lang_data and lang_data['loc']:
                ans, sim = test_single_question(lang_data['loc'], memory, few_shot_examples, test_lang)
                pre_results[f'locality_{test_lang}'] = {
                    'question': lang_data['loc'],
                    'target': lang_data.get('loc_ans', ''),
                    'ans': ans,
                    'similarity': sim
                }

            portability = lang_data.get('portability', {})
            if isinstance(portability, dict) and 'New Question' in portability:
                port_question = portability['New Question']
                ans, sim = test_single_question(port_question, memory, few_shot_examples, test_lang)
                pre_results[f'portability_{test_lang}'] = {
                    'question': port_question,
                    'target': portability.get('New Answer', ''),
                    'ans': ans,
                    'similarity': sim
                }
        
        result_item['pre'] = pre_results

        edit_data = item.get(edit_lang, {})
        edit_question = edit_data.get('src', '')
        edit_answer = edit_data.get('alt', '')
        
        if edit_question and edit_answer:
            memory.add(edit_question, edit_answer)
            print(f"\n[编辑] 注入 {edit_lang.upper()} QA 对到记忆库:")
            print(f"  问题: {edit_question[:80]}...")
            print(f"  答案: {edit_answer}")

        post_results = {}
        
        for test_lang in ['en', 'zh']:
            if test_lang not in item:
                continue
            
            lang_data = item[test_lang]
            few_shot_examples = few_shot_en if test_lang == 'en' else few_shot_zh

            if 'src' in lang_data and lang_data['src']:
                ans, sim = test_single_question(lang_data['src'], memory, few_shot_examples, test_lang)
                post_results[f'src_{test_lang}'] = {
                    'question': lang_data['src'],
                    'target': lang_data.get('alt', ''),
                    'ans': ans,
                    'similarity': sim
                }

            if 'rephrase' in lang_data and lang_data['rephrase']:
                ans, sim = test_single_question(lang_data['rephrase'], memory, few_shot_examples, test_lang)
                post_results[f'rephrase_{test_lang}'] = {
                    'question': lang_data['rephrase'],
                    'target': lang_data.get('alt', ''),
                    'ans': ans,
                    'similarity': sim
                }

            if 'loc' in lang_data and lang_data['loc']:
                ans, sim = test_single_question(lang_data['loc'], memory, few_shot_examples, test_lang)
                post_results[f'locality_{test_lang}'] = {
                    'question': lang_data['loc'],
                    'target': lang_data.get('loc_ans', ''),
                    'ans': ans,
                    'similarity': sim
                }

            portability = lang_data.get('portability', {})
            if isinstance(portability, dict) and 'New Question' in portability:
                port_question = portability['New Question']
                ans, sim = test_single_question(port_question, memory, few_shot_examples, test_lang)
                post_results[f'portability_{test_lang}'] = {
                    'question': port_question,
                    'target': portability.get('New Answer', ''),
                    'ans': ans,
                    'similarity': sim
                }
        
        result_item['post'] = post_results

        memory.clear()
        
        results.append(result_item)

        if (idx + 1) % 10 == 0:
            save_results(results, edit_lang)

    save_results(results, edit_lang)
    
    print(f"\n评估完成！结果已保存")


def save_results(results, edit_lang):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_filename = f"gemini3_api_edit_{edit_lang}_test_all_results.json"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\n结果已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Gemini API Implementation')
    parser.add_argument('--edit_lang', type=str, required=True, choices=['en', 'zh'],
                        help='编辑语言：en 或 zh')
    parser.add_argument('--limit', type=int, default=200,
                        help='测试数据条数限制（默认 200）')
    args = parser.parse_args()
    
    # 加载测试数据
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"找不到测试数据文件: {INPUT_FILE_PATH}")
        return
    
    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载测试数据: {len(data)} 条")

    run_trace_evaluation(data, edit_lang=args.edit_lang, limit=args.limit)


if __name__ == "__main__":
    main()

import json
import os
import time
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F

# ================= 配置区域 =================

# 1. Chinese LLaMA2 模型配置
LLAMA_MODEL_PATH = 'model/chinese-llama2-7b-hf'

# 2. MIKE 配置
# Type 0: Identity, Type 1: Paraphrase, Type 3: Portability 
# 遗传算法优化最佳配置 (200样本分层采样，得分: 84.630)
# 比例: Type 0: 4个, Type 1: 6个, Type 3: 15个
MIKE_ORDER = [3, 3, 3, 0, 3, 3, 3, 3, 1, 0, 1, 1, 1, 0, 3, 1, 3, 3, 3, 3, 3, 0, 1, 3, 3]
NUM_SHOTS = 25
SIMILARITY_MODEL_PATH = 'model/MiniLM-L6-v2'
SIMILARITY_CLASSIFIER_PATH = 'model/6-11-C-1multilingual_similarity_classifier_purexlm_v6A.pth'
SIMILARITY_THRESHOLD = 0.6

# 3. 文件路径
INPUT_FILE_PATH = "data/Bi-ZsRE-data/bizsre_test.json"
RESULTS_DIR = "results"

# 加载 Chinese LLaMA2 模型
print(f" > 正在加载 Chinese LLaMA2 模型: {LLAMA_MODEL_PATH} ...")
try:
    llama_tokenizer = AutoTokenizer.from_pretrained(
        LLAMA_MODEL_PATH,
        use_fast=False,
        trust_remote_code=True
    )
    # 配置tokenizer以适配Chinese-LLaMA2（与B12配置一致）
    llama_tokenizer.truncation_side = "left"
    llama_tokenizer.padding_side = "left"
    
    # 设置pad_token_id为eos_token_id（与B12配置一致，而非默认的unk_token）
    # B12中所有LLaMA模型都使用: tokenizer.pad_token_id = tokenizer.eos_token_id
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id  # pad_token_id = 2 (而非0)
    
    llama_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map='cuda:0',
        trust_remote_code=True
    )
    llama_model.eval()
    
    # 从模型config读取真实的max_position_embeddings（Chinese-LLaMA2是2048，不是4096）
    MAX_CONTEXT_LENGTH = getattr(llama_model.config, "max_position_embeddings", 2048)
    print(f" > Chinese LLaMA2 模型加载完成。使用上下文长度: {MAX_CONTEXT_LENGTH}")
except Exception as e:
    print(f" [错误] 加载 Chinese LLaMA2 模型失败: {e}")
    exit()

# 加载 Embedding 模型（用于few-shot检索）- 已改用XLM-RoBERTa相似度分类器
# print(f" > 正在加载 Embedding 模型: {SIMILARITY_MODEL_PATH} ...")
# try:
#     embedding_tokenizer = AutoTokenizer.from_pretrained(SIMILARITY_MODEL_PATH)
#     embedding_model = AutoModel.from_pretrained(SIMILARITY_MODEL_PATH)
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     embedding_model.to(device)
#     print(" > Embedding 模型加载完成。")
# except Exception as e:
#     print(f" [错误] 加载 Embedding 模型失败: {e}")
#     exit()

# 设置设备（原本在embedding_model加载时设置）
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 加载相似度分类器（用于记忆库检索）- 使用B12的SERAC实现
print(f" > 正在加载相似度分类器: {SIMILARITY_CLASSIFIER_PATH} ...")
try:
    from transformers import XLMRobertaTokenizer, XLMRobertaModel
    # 使用本地xlm-roberta-base模型路径（与B12一致）
    XLM_MODEL_PATH = '../B12-fewshot-clear-all/model/xlm-roberta-base'
    classifier_tokenizer = XLMRobertaTokenizer.from_pretrained(XLM_MODEL_PATH)
    classifier_backbone = XLMRobertaModel.from_pretrained(XLM_MODEL_PATH)
    
    # 加载微调后的XLM权重（与B12 SERAC一致）
    checkpoint = torch.load(SIMILARITY_CLASSIFIER_PATH, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        classifier_backbone.load_state_dict(checkpoint['model_state_dict'])
    elif 'classifier' in checkpoint:
        classifier_backbone.load_state_dict(checkpoint['classifier'], strict=False)
    else:
        raise ValueError(f"Archive missing classifier weights keys: {list(checkpoint.keys())}")
    
    classifier_backbone.to(device)
    classifier_backbone.eval()
    print(" > 相似度分类器加载完成。")
except Exception as e:
    print(f" [错误] 加载相似度分类器失败: {e}")
    print(" > MIKE方法需要相似度分类器，程序退出")
    exit()

def get_embedding(text_list):
    """使用XLM-RoBERTa相似度分类器计算embedding（用于few-shot检索）"""
    encoded_input = classifier_tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = classifier_backbone(**encoded_input)
    token_embeddings = model_output.last_hidden_state
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def get_similarity_score(question, memory_question):
    """使用相似度分类器计算两个问题的相似度分数（参考B12 SERAC实现）"""
    try:
        # 使用Mean Pooling计算embedding（与B12训练时一致）
        cls_ctx_input = classifier_tokenizer([memory_question], return_tensors="pt", padding=True).to(device)
        cls_main_input = classifier_tokenizer([question], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # 获取模型输出
            ctx_outputs = classifier_backbone(**cls_ctx_input)
            main_outputs = classifier_backbone(**cls_main_input)
            
            # Mean Pooling for ctx (memory)
            ctx_token_embeddings = ctx_outputs.last_hidden_state
            ctx_attention_mask = cls_ctx_input['attention_mask']
            ctx_input_mask_expanded = ctx_attention_mask.unsqueeze(-1).expand(ctx_token_embeddings.size()).float()
            ctx_sum_embeddings = torch.sum(ctx_token_embeddings * ctx_input_mask_expanded, 1)
            ctx_sum_mask = torch.clamp(ctx_input_mask_expanded.sum(1), min=1e-9)
            ctx_mean_embeddings = ctx_sum_embeddings / ctx_sum_mask
            
            # Mean Pooling for main (query)
            main_token_embeddings = main_outputs.last_hidden_state
            main_attention_mask = cls_main_input['attention_mask']
            main_input_mask_expanded = main_attention_mask.unsqueeze(-1).expand(main_token_embeddings.size()).float()
            main_sum_embeddings = torch.sum(main_token_embeddings * main_input_mask_expanded, 1)
            main_sum_mask = torch.clamp(main_input_mask_expanded.sum(1), min=1e-9)
            main_mean_embeddings = main_sum_embeddings / main_sum_mask
            
            # 计算余弦相似度（与B12训练时一致）
            cos_sim = (ctx_mean_embeddings * main_mean_embeddings).sum(-1) / (
                ctx_mean_embeddings.norm(2, -1) * main_mean_embeddings.norm(2, -1)
            )
            
            # Clamp到[-1, 1]范围
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            
            # 归一化到[0, 1]范围（与B12训练时一致）
            normalized_sim = (cos_sim + 1) / 2
            normalized_sim = torch.clamp(normalized_sim, 0.0, 1.0)
            
            return normalized_sim.item()
            
    except Exception as e:
        print(f"相似度计算错误: {e}")
        return 0.0

class MemoryBank:
    """记忆库：存储编辑的QA对"""
    def __init__(self):
        self.memories = []  # 存储 {'question': str, 'answer': str, 'lang': str}
    
    def add_memory(self, question, answer, lang='en'):
        """添加一个QA对到记忆库"""
        self.memories.append({
            'question': question,
            'answer': answer,
            'lang': lang
        })
    
    def search_memory(self, query_question, threshold=0.6):
        """在记忆库中搜索与查询问题相似的记忆
        
        Returns:
            (similarity_score, memory_dict) 如果找到相似度>threshold的记忆
            (0.0, None) 如果没有找到
        """
        if not self.memories:
            return 0.0, None
        
        max_similarity = 0.0
        best_memory = None
        
        for memory in self.memories:
            similarity = get_similarity_score(query_question, memory['question'])
            if similarity > max_similarity:
                max_similarity = similarity
                best_memory = memory
        
        if max_similarity >= threshold:
            return max_similarity, best_memory
        else:
            return max_similarity, None
    
    def clear(self):
        """清空记忆库"""
        self.memories = []

def select_demonstrations_mixed(all_data, current_idx, current_query, k=32):
    """选择few-shot demonstrations（与IKE相同）"""
    candidate_texts = []
    candidate_meta = [] 
    
    # 1. 构建混合候选池
    for idx, item in enumerate(all_data):
        if idx == current_idx: continue 
        
        if 'en' in item and item['en'].get('src'):
            candidate_texts.append(item['en']['src'])
            candidate_meta.append({'idx': idx, 'lang': 'en'})
            
        if 'zh' in item and item['zh'].get('src'):
            candidate_texts.append(item['zh']['src'])
            candidate_meta.append({'idx': idx, 'lang': 'zh'})

    if not candidate_texts:
        return []

    # 2. 计算相似度
    query_embedding = get_embedding([current_query]) 
    
    # 简单分批处理防止可能的 OOM
    batch_size = 512
    all_embeddings = []
    for i in range(0, len(candidate_texts), batch_size):
        batch_texts = candidate_texts[i:i+batch_size]
        all_embeddings.append(get_embedding(batch_texts))
    
    if all_embeddings:
        candidate_embeddings = torch.cat(all_embeddings, dim=0)
        cosine_scores = torch.mm(query_embedding, candidate_embeddings.transpose(0, 1))[0]
        
        # 3. 直接取 Top-K
        k = min(k, len(candidate_texts))
        top_results = torch.topk(cosine_scores, k=k)
        top_indices = top_results.indices.tolist()
        
        selected_demos = []
        for pool_idx in top_indices:
            meta = candidate_meta[pool_idx]
            original_idx = meta['idx']
            selected_demos.append({
                'item': all_data[original_idx], 
                'use_lang': meta['lang']   
            })
        return selected_demos
    return []

def construct_mike_prompt(demonstrations, memory_qa, current_item, edit_lang='en', test_lang='en', custom_test_question=None):
    """构建MIKE prompt：few-shot + 记忆QA + 问题
    
    Args:
        demonstrations: few-shot demonstrations
        memory_qa: 记忆库中检索到的QA对 {'question': str, 'answer': str}
        current_item: 当前测试项
        edit_lang: 编辑语言
        test_lang: 测试语言
        custom_test_question: 自定义测试问题
    """
    # 1. 获取编辑语言的目标信息
    edit_data = current_item.get(edit_lang, {})
    edit_src = edit_data.get('src') 
    edit_alt = edit_data.get('alt')
    
    context_parts = []
    current_order = MIKE_ORDER[:NUM_SHOTS]
    shuffled_order = current_order.copy()
    random.shuffle(shuffled_order)
    
    demo_pointer = 0
    
    # 2. 构建few-shot demonstrations（去除Type 2: Locality）
    for type_code in shuffled_order:
        if demo_pointer >= len(demonstrations):
            continue
        
        demo_info = demonstrations[demo_pointer]
        demo_pointer += 1
        
        demo_lang = demo_info['use_lang']
        demo_data = demo_info['item'].get(demo_lang, {})
        
        if not demo_data:
            continue
        
        d_src = demo_data.get('src')
        d_alt = demo_data.get('alt')
        d_rephrase = demo_data.get('rephrase')
        d_portability = demo_data.get('portability', {})
        d_port_question = d_portability.get('New Question', '') if isinstance(d_portability, dict) else ''
        d_port_answer = d_portability.get('New Answer', '') if isinstance(d_portability, dict) else ''
        
        # New Fact 始终使用该 demonstration 的 src + alt
        demo_new_fact = f"New Fact: Question: {d_src} Answer: {d_alt}"
        prompt_str = ""
        
        if type_code == 0:  # Identity - src + alt
            prompt_str = f"Prompt: Question: {d_src} Answer: {d_alt}"
            
        elif type_code == 1:  # Paraphrase - rephrase + alt
            if d_rephrase:
                prompt_str = f"Prompt: Question: {d_rephrase} Answer: {d_alt}"
            else:
                prompt_str = f"Prompt: Question: {d_src} Answer: {d_alt}"
                
        elif type_code == 3:  # Portability - portability question + answer
            if d_port_question and d_port_answer:
                prompt_str = f"Prompt: Question: {d_port_question} Answer: {d_port_answer}"
            else:
                continue

        if prompt_str:
            context_parts.append(f"{demo_new_fact}\n{prompt_str}")

    # 3. 添加记忆库中的QA对
    memory_fact_str = f"New Fact: Question: {memory_qa['question']} Answer: {memory_qa['answer']}"
    
    # 4. 获取测试问题
    if custom_test_question:
        test_question = custom_test_question
    else:
        test_data = current_item.get(test_lang, {})
        test_question = test_data.get('src')
    
    # 5. 组装完整prompt
    full_context = "\n\n".join(context_parts)
    final_instruction = f"{full_context}\n\n{memory_fact_str}\nPrompt: Question: {test_question} Answer:"
    
    return final_instruction

def construct_simple_prompt(question):
    """构建简单的问答prompt（相似度<0.6时使用）"""
    return f"Question: {question} Answer:"

def query_model_with_target(prompt, target):
    """使用Chinese-LLaMA2模型进行token by token推理"""
    try:
        # 处理target，确保有前导空格
        target_text = target if (isinstance(target, str) and target[:1].isspace()) else f" {target}"
        
        # 对target进行tokenize
        target_ids = llama_tokenizer(target_text, add_special_tokens=False, return_tensors='pt')["input_ids"].to(llama_model.device)
        
        if target_ids.size(1) == 0:
            return "", target
        
        # 构建完整的输入（prompt + target）
        full_input = prompt + target_text
        encodings = llama_tokenizer(full_input, return_tensors='pt', truncation=True, max_length=MAX_CONTEXT_LENGTH)
        input_ids = encodings['input_ids'].to(llama_model.device)
        attention_mask = encodings['attention_mask'].to(llama_model.device)
        
        # 一次性推理
        with torch.no_grad():
            outputs = llama_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        # 提取对应位置的预测tokens
        L = target_ids.size(1)
        ans = torch.argmax(logits, dim=-1)[:, -L-1:-1].squeeze()
        
        # 解码预测的tokens
        ans_ids = ans.detach().cpu().numpy().tolist()
        if not isinstance(ans_ids, list):
            ans_ids = [ans_ids]
        
        predicted_answer = llama_tokenizer.decode(ans_ids, skip_special_tokens=True).strip()
        
        # 清理常见的答案前缀（提高EM分数）
        prefixes_to_remove = [": ", ":", " ", "：", "： "]
        for prefix in prefixes_to_remove:
            if predicted_answer.startswith(prefix):
                predicted_answer = predicted_answer[len(prefix):].strip()
                break
        
        return predicted_answer, target
        
    except Exception as e:
        print(f"模型推理错误: {e}")
        return "", target

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edit_lang', type=str, required=True, choices=['en', 'zh'], 
                        help='编辑语言：en 或 zh')
    parser.add_argument('--limit', type=int, default=-1, 
                        help='测试数据量限制，默认全部数据')
    args = parser.parse_args()
    
    edit_lang = args.edit_lang
    LIMIT = args.limit
    
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"找不到文件: {INPUT_FILE_PATH}")
        return

    with open(INPUT_FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_data = data[:LIMIT] if LIMIT > 0 else data
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 输出文件路径
    output_filename = f"chinese_llama2_7b_mike_edit_{edit_lang}_test_results.json"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    print(f"="*60)
    print(f"MIKE方法测试 (Chinese-LLaMA2)")
    print(f"编辑语言: {edit_lang}")
    print(f"相似度阈值: {SIMILARITY_THRESHOLD}")
    print(f"计划处理 {len(test_data)} 条数据")
    print(f"结果保存至: {output_path}")
    print(f"="*60)
    
    # 断点续传
    results = []
    start_index = 0
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            start_index = len(results)
            print(f" > 发现已存在的进度文件，包含 {start_index} 条结果。")
            if start_index >= len(test_data):
                print(" > 所有数据似乎已处理完毕。")
                return
            else:
                print(f" > 将从第 {start_index + 1} 条开始继续运行...")
        except json.JSONDecodeError:
            print(" > [警告] 现有输出文件损坏或为空，将从头开始。")
            results = []

    # 创建记忆库
    memory_bank = MemoryBank()
    
    # Token长度统计
    prompt_token_lengths = []

    for idx, item in tqdm(enumerate(test_data), total=len(test_data)):
        if idx < start_index:
            continue
        
        start_time = time.time()
        
        # 获取数据
        edit_data = item.get(edit_lang, {})
        en_data = item.get('en', {})
        zh_data = item.get('zh', {})

        # 构建结果项基础结构
        result_item = {
            'pre': {},
            'post': {},
            'case_id': idx,
            'requested_rewrite': {
                'prompt': edit_data.get('src', ''),
                'target_new_en': en_data.get('alt', ''),
                'target_new_zh': zh_data.get('alt', ''),
                'ground_truth': ''
            }
        }

        # ========== PRE-EDIT 推理（不使用记忆库）==========
        # 1. Rewrite accuracy
        pre_prompt_rewrite = construct_simple_prompt(edit_data.get('src', ''))
        pre_ans_rewrite, pre_target_rewrite = query_model_with_target(pre_prompt_rewrite, edit_data.get('alt', ''))
        result_item['pre']['rewrite_acc'] = {'ans': pre_ans_rewrite, 'target': pre_target_rewrite}
        
        # 2. Rephrase accuracy
        if en_data.get('rephrase'):
            pre_prompt_rephrase_en = construct_simple_prompt(en_data.get('rephrase', ''))
            pre_ans_rephrase_en, pre_target_rephrase_en = query_model_with_target(pre_prompt_rephrase_en, en_data.get('alt', ''))
            result_item['pre']['rephrase_acc_en'] = {'ans': pre_ans_rephrase_en, 'target': pre_target_rephrase_en}
        else:
            result_item['pre']['rephrase_acc_en'] = {}
            
        if zh_data.get('rephrase'):
            pre_prompt_rephrase_zh = construct_simple_prompt(zh_data.get('rephrase', ''))
            pre_ans_rephrase_zh, pre_target_rephrase_zh = query_model_with_target(pre_prompt_rephrase_zh, zh_data.get('alt', ''))
            result_item['pre']['rephrase_acc_zh'] = {'ans': pre_ans_rephrase_zh, 'target': pre_target_rephrase_zh}
        else:
            result_item['pre']['rephrase_acc_zh'] = {}
        
        # 3. Locality
        if en_data.get('loc') and en_data.get('loc_ans'):
            pre_prompt_loc_en = construct_simple_prompt(en_data.get('loc', ''))
            pre_ans_loc_en, pre_target_loc_en = query_model_with_target(pre_prompt_loc_en, en_data.get('loc_ans', ''))
            result_item['pre']['locality_en'] = {'neighborhood_output_en': {'ans': pre_ans_loc_en, 'target': pre_target_loc_en}}
        else:
            result_item['pre']['locality_en'] = {'neighborhood_output_en': {}}
            
        if zh_data.get('loc') and zh_data.get('loc_ans'):
            pre_prompt_loc_zh = construct_simple_prompt(zh_data.get('loc', ''))
            pre_ans_loc_zh, pre_target_loc_zh = query_model_with_target(pre_prompt_loc_zh, zh_data.get('loc_ans', ''))
            result_item['pre']['locality_zh'] = {'neighborhood_output_zh': {'ans': pre_ans_loc_zh, 'target': pre_target_loc_zh}}
        else:
            result_item['pre']['locality_zh'] = {'neighborhood_output_zh': {}}
        
        # 4. Portability
        en_portability = en_data.get('portability', {})
        if isinstance(en_portability, dict) and en_portability.get('New Question') and en_portability.get('New Answer'):
            pre_prompt_port_en = construct_simple_prompt(en_portability.get('New Question', ''))
            pre_ans_port_en, pre_target_port_en = query_model_with_target(pre_prompt_port_en, en_portability.get('New Answer', ''))
            result_item['pre']['portability_en'] = {'one_hop_acc_en': {'ans': pre_ans_port_en, 'target': pre_target_port_en}}
        else:
            result_item['pre']['portability_en'] = {'one_hop_acc_en': {}}
            
        zh_portability = zh_data.get('portability', {})
        if isinstance(zh_portability, dict) and zh_portability.get('New Question') and zh_portability.get('New Answer'):
            pre_prompt_port_zh = construct_simple_prompt(zh_portability.get('New Question', ''))
            pre_ans_port_zh, pre_target_port_zh = query_model_with_target(pre_prompt_port_zh, zh_portability.get('New Answer', ''))
            result_item['pre']['portability_zh'] = {'one_hop_acc_en': {'ans': pre_ans_port_zh, 'target': pre_target_port_zh}}
        else:
            result_item['pre']['portability_zh'] = {'one_hop_acc_en': {}}

        # ========== POST-EDIT 推理（使用MIKE方法）==========
        # 清空记忆库（每次编辑后都清空，只影响当前7个问题）
        memory_bank.clear()
        
        # 将当前编辑的QA对添加到记忆库
        memory_bank.add_memory(
            question=edit_data.get('src', ''),
            answer=edit_data.get('alt', ''),
            lang=edit_lang
        )
        
        # 1. Rewrite accuracy - 检索记忆库
        similarity, memory = memory_bank.search_memory(edit_data.get('src', ''), threshold=SIMILARITY_THRESHOLD)
        
        if memory is not None:
            # 相似度 >= 0.6，使用MIKE prompt
            demos_rewrite = select_demonstrations_mixed(data, idx, edit_data.get('src', ''), k=NUM_SHOTS)
            post_prompt_rewrite = construct_mike_prompt(demos_rewrite, memory, item, edit_lang=edit_lang, test_lang=edit_lang)
            # 记录prompt长度
            prompt_tokens = len(llama_tokenizer(post_prompt_rewrite, add_special_tokens=True)["input_ids"])
            prompt_token_lengths.append(prompt_tokens)
        else:
            # 相似度 < 0.6，直接问
            post_prompt_rewrite = construct_simple_prompt(edit_data.get('src', ''))
        
        post_ans_rewrite, post_target_rewrite = query_model_with_target(post_prompt_rewrite, edit_data.get('alt', ''))
        result_item['post']['rewrite_acc'] = {'ans': post_ans_rewrite, 'target': post_target_rewrite}
        
        # 2. Rephrase accuracy
        if en_data.get('rephrase'):
            similarity, memory = memory_bank.search_memory(en_data.get('rephrase', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_rephrase_en = select_demonstrations_mixed(data, idx, en_data.get('rephrase', ''), k=NUM_SHOTS)
                post_prompt_rephrase_en = construct_mike_prompt(demos_rephrase_en, memory, item, edit_lang=edit_lang, test_lang='en', custom_test_question=en_data.get('rephrase', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_rephrase_en, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_rephrase_en = construct_simple_prompt(en_data.get('rephrase', ''))
            post_ans_rephrase_en, post_target_rephrase_en = query_model_with_target(post_prompt_rephrase_en, en_data.get('alt', ''))
            result_item['post']['rephrase_acc_en'] = {'ans': post_ans_rephrase_en, 'target': post_target_rephrase_en}
        else:
            result_item['post']['rephrase_acc_en'] = {}
            
        if zh_data.get('rephrase'):
            similarity, memory = memory_bank.search_memory(zh_data.get('rephrase', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_rephrase_zh = select_demonstrations_mixed(data, idx, zh_data.get('rephrase', ''), k=NUM_SHOTS)
                post_prompt_rephrase_zh = construct_mike_prompt(demos_rephrase_zh, memory, item, edit_lang=edit_lang, test_lang='zh', custom_test_question=zh_data.get('rephrase', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_rephrase_zh, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_rephrase_zh = construct_simple_prompt(zh_data.get('rephrase', ''))
            post_ans_rephrase_zh, post_target_rephrase_zh = query_model_with_target(post_prompt_rephrase_zh, zh_data.get('alt', ''))
            result_item['post']['rephrase_acc_zh'] = {'ans': post_ans_rephrase_zh, 'target': post_target_rephrase_zh}
        else:
            result_item['post']['rephrase_acc_zh'] = {}
        
        # 3. Locality
        if en_data.get('loc') and en_data.get('loc_ans'):
            similarity, memory = memory_bank.search_memory(en_data.get('loc', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_loc_en = select_demonstrations_mixed(data, idx, en_data.get('loc', ''), k=NUM_SHOTS)
                post_prompt_loc_en = construct_mike_prompt(demos_loc_en, memory, item, edit_lang=edit_lang, test_lang='en', custom_test_question=en_data.get('loc', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_loc_en, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_loc_en = construct_simple_prompt(en_data.get('loc', ''))
            post_ans_loc_en, post_target_loc_en = query_model_with_target(post_prompt_loc_en, en_data.get('loc_ans', ''))
            result_item['post']['locality_en'] = {'neighborhood_output_en': {'ans': post_ans_loc_en, 'target': post_target_loc_en}}
        else:
            result_item['post']['locality_en'] = {'neighborhood_output_en': {}}
            
        if zh_data.get('loc') and zh_data.get('loc_ans'):
            similarity, memory = memory_bank.search_memory(zh_data.get('loc', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_loc_zh = select_demonstrations_mixed(data, idx, zh_data.get('loc', ''), k=NUM_SHOTS)
                post_prompt_loc_zh = construct_mike_prompt(demos_loc_zh, memory, item, edit_lang=edit_lang, test_lang='zh', custom_test_question=zh_data.get('loc', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_loc_zh, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_loc_zh = construct_simple_prompt(zh_data.get('loc', ''))
            post_ans_loc_zh, post_target_loc_zh = query_model_with_target(post_prompt_loc_zh, zh_data.get('loc_ans', ''))
            result_item['post']['locality_zh'] = {'neighborhood_output_zh': {'ans': post_ans_loc_zh, 'target': post_target_loc_zh}}
        else:
            result_item['post']['locality_zh'] = {'neighborhood_output_zh': {}}
        
        # 4. Portability
        if isinstance(en_portability, dict) and en_portability.get('New Question') and en_portability.get('New Answer'):
            similarity, memory = memory_bank.search_memory(en_portability.get('New Question', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_port_en = select_demonstrations_mixed(data, idx, en_portability.get('New Question', ''), k=NUM_SHOTS)
                post_prompt_port_en = construct_mike_prompt(demos_port_en, memory, item, edit_lang=edit_lang, test_lang='en', custom_test_question=en_portability.get('New Question', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_port_en, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_port_en = construct_simple_prompt(en_portability.get('New Question', ''))
            post_ans_port_en, post_target_port_en = query_model_with_target(post_prompt_port_en, en_portability.get('New Answer', ''))
            result_item['post']['portability_en'] = {'one_hop_acc_en': {'ans': post_ans_port_en, 'target': post_target_port_en}}
        else:
            result_item['post']['portability_en'] = {'one_hop_acc_en': {}}
            
        if isinstance(zh_portability, dict) and zh_portability.get('New Question') and zh_portability.get('New Answer'):
            similarity, memory = memory_bank.search_memory(zh_portability.get('New Question', ''), threshold=SIMILARITY_THRESHOLD)
            if memory is not None:
                demos_port_zh = select_demonstrations_mixed(data, idx, zh_portability.get('New Question', ''), k=NUM_SHOTS)
                post_prompt_port_zh = construct_mike_prompt(demos_port_zh, memory, item, edit_lang=edit_lang, test_lang='zh', custom_test_question=zh_portability.get('New Question', ''))
                prompt_tokens = len(llama_tokenizer(post_prompt_port_zh, add_special_tokens=True)["input_ids"])
                prompt_token_lengths.append(prompt_tokens)
            else:
                post_prompt_port_zh = construct_simple_prompt(zh_portability.get('New Question', ''))
            post_ans_port_zh, post_target_port_zh = query_model_with_target(post_prompt_port_zh, zh_portability.get('New Answer', ''))
            result_item['post']['portability_zh'] = {'one_hop_acc_en': {'ans': post_ans_port_zh, 'target': post_target_port_zh}}
        else:
            result_item['post']['portability_zh'] = {'one_hop_acc_en': {}}

        results.append(result_item)
        
        # 每10条保存一次
        if (idx + 1) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"处理第 {idx + 1} 条数据，耗时 {time.time() - start_time:.2f} 秒")
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 输出token长度统计
    if prompt_token_lengths:
        import numpy as np
        lengths = np.array(prompt_token_lengths)
        print(f"\n{'='*60}")
        print(f"Token长度统计 (共 {len(lengths)} 个MIKE prompts):")
        print(f"  最小值: {lengths.min()}")
        print(f"  最大值: {lengths.max()}")
        print(f"  平均值: {lengths.mean():.1f}")
        print(f"  中位数: {np.median(lengths):.1f}")
        print(f"  标准差: {lengths.std():.1f}")
        print(f"  25分位: {np.percentile(lengths, 25):.1f}")
        print(f"  75分位: {np.percentile(lengths, 75):.1f}")
        print(f"  95分位: {np.percentile(lengths, 95):.1f}")
        print(f"  99分位: {np.percentile(lengths, 99):.1f}")
        print(f"  超过1500的: {(lengths > 1500).sum()} ({(lengths > 1500).sum() / len(lengths) * 100:.1f}%)")
        print(f"  超过1800的: {(lengths > 1800).sum()} ({(lengths > 1800).sum() / len(lengths) * 100:.1f}%)")
        print(f"  超过2000的: {(lengths > 2000).sum()} ({(lengths > 2000).sum() / len(lengths) * 100:.1f}%)")
        print(f"  最大上下文: {MAX_CONTEXT_LENGTH}")
        print(f"{'='*60}")
        
    print(f"\n完成！结果已保存至: {output_path}")
    
if __name__ == "__main__":
    random.seed(42)
    main()

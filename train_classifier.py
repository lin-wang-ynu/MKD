import json
import random
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import copy
from collections import defaultdict


# ========================================
# 语言组合编码（保留，用于数据生成标签）
# ========================================
def encode_language_combination(edit_lang, test_lang):
    lang_mapping = {
        ("en", "en"): 0,  # 英英
        ("zh", "zh"): 1,  # 中中
        ("en", "zh"): 2,  # 英中
        ("zh", "en"): 3,  # 中英
    }
    return lang_mapping.get((edit_lang, test_lang), 0)


# ========================================
# 纯 XLM 相似度模型（不再使用额外的语言 / portability 嵌入）
# 与 SERAC 推理端结构严格对齐：底层 AutoModel + AutoTokenizer，
# 相似度使用归一化余弦相似度 [0, 1]。
# ========================================
class PureXLMSimilarityClassifier(nn.Module):
    def __init__(self, model_name: str = "model/xlm-roberta-base"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _encode(self, questions):
        """将一批问题编码为句向量表示，使用 Mean Pooling 方法。"""
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # 使用 Mean Pooling（考虑 attention mask）
        # 这是 XLM-RoBERTa 和相似度任务的最佳实践
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        attention_mask = inputs['attention_mask']  # [batch_size, seq_len]
        
        # 扩展 attention mask 的维度以匹配 token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 对所有 token 的 embeddings 求和（忽略 padding）
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # 计算实际 token 数量（避免除零）
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # 计算平均值
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def forward(self, edit_questions, test_questions):
        """
        返回两组问题的向量表示：
        - edit_questions: 作为“编辑问题 / 记忆问题”
        - test_questions: 作为“测试问题”
        """
        edit_embeddings = self._encode(edit_questions)
        test_embeddings = self._encode(test_questions)
        return edit_embeddings, test_embeddings


# ========================================
# 训练数据集（保持原样）
# ========================================
class SimilarityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ========================================
# 自定义 collate 函数（portability_flag 统一补 0，虽然后续不再使用）
# ========================================
def collate_fn(batch):
    for item in batch:
        if "portability_flag" not in item:
            item["portability_flag"] = 0
    return batch


# ========================================
# 数据生成器（保持原有规则，用标签控制 original / generalization / portability / local 等）
# ========================================
class MultilingualDataGenerator:
    def __init__(self):
        self.similarity_rules = {
            "original": 1.0,
            "generalization": 0.9,
            "portability": 0.85,
            "local": 0.2,
            "unrelated": 0.3,
            "local_anchor": 1.0,
        }

        # 两个锚点比例
        self.anchor_ratios = {
            "edit_anchor": 0.6,  # 锚点1：编辑问题
            "local_anchor": 0.4,  # 锚点2：局部性问题
        }

        # edit_anchor 内部正样本比例
        self.edit_positive_ratios = {
            "original": 0.4,
            "generalization": 0.4,
            "portability": 0.2,
        }

        # 跨语言样本比例提升
        self.cross_lang_ratio = 0.6  # 60% 样本为跨语言

    def generate_training_data(self, en_data, zh_data, max_samples_per_type=10000):
        training_data = []
        for edit_lang, test_lang in [("en", "en"), ("zh", "zh"), ("en", "zh"), ("zh", "en")]:
            print(f"生成 {edit_lang}_{test_lang} 训练数据...")
            is_cross = edit_lang != test_lang
            adjusted_max = int(
                max_samples_per_type * (self.cross_lang_ratio if is_cross else (1 - self.cross_lang_ratio / 3))
            )
            combo_data = self.generate_language_combo_data(
                en_data, zh_data, edit_lang, test_lang, adjusted_max
            )
            training_data.extend(combo_data)
            print(f"{edit_lang}_{test_lang} 生成了 {len(combo_data)} 个样本")
        random.shuffle(training_data)

        output_path = "data/generated_training_data_contrastive_v6A.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        print(f"训练样本已保存到 {output_path}")
        return training_data

    def generate_language_combo_data(self, en_data, zh_data, edit_lang, test_lang, max_samples_per_type):
        edit_data = en_data if edit_lang == "en" else zh_data
        test_data = en_data if test_lang == "en" else zh_data

        total_target = max_samples_per_type
        edit_anchor_count = int(total_target * self.anchor_ratios["edit_anchor"])
        local_anchor_count = int(total_target * self.anchor_ratios["local_anchor"])

        original_count = int(edit_anchor_count * self.edit_positive_ratios["original"])
        generalization_count = int(edit_anchor_count * self.edit_positive_ratios["generalization"])
        portability_count = int(edit_anchor_count * self.edit_positive_ratios["portability"])

        counts = {"original": 0, "generalization": 0, "portability": 0, "local_anchor": 0}

        # unrelated 采样池
        subject_to_samples = defaultdict(list)
        for idx, sample in enumerate(test_data):
            subject_to_samples[sample["subject"]].append((idx, sample))
        all_subjects = list(subject_to_samples.keys())

        combo_data = []

        while (
            counts["original"] < original_count
            or counts["generalization"] < generalization_count
            or counts["portability"] < portability_count
            or counts["local_anchor"] < local_anchor_count
        ):
            i = random.randint(0, len(edit_data) - 1)
            edit_sample = edit_data[i]
            test_sample = test_data[i]

            edit_q = "Q: " + edit_sample["src"].replace("nq question: ", "").strip()
            edit_ans = edit_sample["alt"]
            edit_subj = edit_sample["subject"]

            src = test_sample["src"].replace("nq question: ", "").strip()
            rephrase = test_sample["rephrase"].replace("nq question: ", "").strip()
            loc = test_sample["loc"].replace("nq question: ", "").strip()
            # 由于训练集中没有 portability 显式字段，使用 rephrase 作为默认 portability 问题
            port_q = (
                test_sample.get("portability", {}).get("New Question", rephrase).replace("nq question: ", "").strip()
            )

            lang_code = encode_language_combination(edit_lang, test_lang)

            # 锚点1：edit_anchor
            if counts["original"] < original_count:
                sample = self._make_edit_anchor_sample(
                    edit_q,
                    edit_ans,
                    edit_subj,
                    edit_lang,
                    src,
                    test_lang,
                    "original",
                    lang_code,
                    loc,
                    subject_to_samples,
                    all_subjects,
                    portability_flag=0,
                )
                combo_data.append(sample)
                counts["original"] += 1

            if counts["generalization"] < generalization_count:
                sample = self._make_edit_anchor_sample(
                    edit_q,
                    edit_ans,
                    edit_subj,
                    edit_lang,
                    rephrase,
                    test_lang,
                    "generalization",
                    lang_code,
                    loc,
                    subject_to_samples,
                    all_subjects,
                    portability_flag=0,
                )
                combo_data.append(sample)
                counts["generalization"] += 1

            if counts["portability"] < portability_count:
                sample = self._make_edit_anchor_sample(
                    edit_q,
                    edit_ans,
                    edit_subj,
                    edit_lang,
                    port_q,
                    test_lang,
                    "portability",
                    lang_code,
                    loc,
                    subject_to_samples,
                    all_subjects,
                    portability_flag=1,
                )
                combo_data.append(sample)
                counts["portability"] += 1

            # 锚点2：local_anchor
            if counts["local_anchor"] < local_anchor_count:
                sample = self._make_local_anchor_sample(loc, test_lang, src, rephrase, port_q)
                combo_data.append(sample)
                counts["local_anchor"] += 1

            if len(combo_data) > total_target * 3:
                print(f"警告：{edit_lang}_{test_lang} 样本过多，提前终止")
                break

        return combo_data

    def _make_edit_anchor_sample(
        self,
        edit_q,
        edit_ans,
        edit_subj,
        edit_lang,
        pos_q,
        test_lang,
        pos_type,
        lang_code,
        loc,
        subject_to_samples,
        all_subjects,
        portability_flag=0,
    ):
        anchor = {"question": edit_q, "answer": edit_ans, "subject": edit_subj, "lang": edit_lang}
        positive = {"question": pos_q, "lang": test_lang}

        # 负样本1：local
        neg_local = {"question": loc, "lang": test_lang, "neg_type": "local"}

        # 负样本2：unrelated
        other_subjs = [s for s in all_subjects if s != edit_subj]
        if other_subjs:
            other_subj = random.choice(other_subjs)
            _, other_s = random.choice(subject_to_samples[other_subj])
            unrel_q = other_s["loc"].replace("nq question: ", "").strip()
        else:
            unrel_q = loc
        neg_unrelated = {"question": unrel_q, "lang": test_lang, "neg_type": "unrelated"}

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": [neg_local, neg_unrelated],
            "similarity_label": self.similarity_rules[pos_type],
            "similarity_label_negative": self.similarity_rules["unrelated"],
            "sample_type": f"{pos_type}_positive",
            "language_combination": lang_code,
            "portability_flag": portability_flag,
        }

    def _make_local_anchor_sample(self, loc, test_lang, src, rephrase, port_q):
        anchor = {"question": "Q: " + loc, "lang": test_lang}
        positive = {"question": loc, "lang": test_lang}
        neg_q = random.choice([src, rephrase, port_q])
        negative = {"question": neg_q, "lang": test_lang}
        return {
            "anchor": anchor,
            "positive": positive,
            "negative": [negative],
            "similarity_label": 1.0,
            "similarity_label_negative": 0.2,
            "sample_type": "local_anchor",
            "language_combination": encode_language_combination(test_lang, test_lang),
            "portability_flag": 0,
        }


# ========================================
# 分类器训练器（仅使用纯 XLM + 余弦相似度）
# ========================================
class MemorylessClassifierTrainer:
    def __init__(self, classifier: PureXLMSimilarityClassifier, optimizer):
        self.classifier = classifier
        self.optimizer = optimizer
        self.mse_loss = nn.MSELoss()

    def margin_mse(self, pred, target, margin=0.05, power=2):
        """
        仅当 |pred - target| > margin 时产生损失；且偏差越大惩罚越高。
        """
        diff = torch.abs(pred - target)
        excess = torch.relu(diff - margin)
        return (excess ** power).mean()

    def train_without_memory(
        self,
        train_loader,
        val_loader=None,
        epochs=20,
        device="cuda",
        patience=3,
        min_delta=0.001,
        temperature=0.07,
        regression_weight=0.8,
    ):
        self.classifier.to(device)
        best_ratio = 0.0  # 改为跟踪 below_05_ratio，越大越好
        patience_counter = 0
        best_model_weights = None
        similarity_distributions = None

        for epoch in range(epochs):
            self.classifier.train()
            total_loss = 0
            correct_predictions = 0
            total_predictions = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                anchor_questions = [item["anchor"]["question"] for item in batch]
                positive_questions = [item["positive"]["question"] for item in batch]
                negative_questions_list = [item["negative"] for item in batch]
                sim_labels = torch.tensor([item["similarity_label"] for item in batch], dtype=torch.float32).to(device)

                anchor_embeds, positive_embeds = self.classifier(anchor_questions, positive_questions)

                contrastive_loss = self.info_nce_loss(
                    anchor_embeds, positive_embeds, negative_questions_list, anchor_questions, device, temperature
                )

                # 与推理保持一致：归一化余弦相似度到 [0, 1]
                pos_sim_raw = torch.nn.functional.cosine_similarity(anchor_embeds, positive_embeds, dim=-1)
                pos_sim = torch.clamp((pos_sim_raw + 1) / 2, 0.0, 1.0)
                regression_loss = self.margin_mse(pos_sim, sim_labels, margin=0.05, power=1)

                # 负样本相似度约束（local 更强约束）
                neg_regression_loss = 0.0
                num_negs = 0
                local_weight = 3.0
                local_margin = 0.15
                hinge_weight = 1.0
                for i, negs in enumerate(negative_questions_list):
                    if negs:
                        neg_q = [n["question"] for n in negs]
                        neg_types = [n.get("neg_type", "") for n in negs]
                        # 使用相同的 anchor 文本，分别与多个负样本计算向量
                        _, neg_embeds = self.classifier([anchor_questions[i]] * len(neg_q), neg_q)
                        neg_sim_raw = torch.nn.functional.cosine_similarity(
                            anchor_embeds[i : i + 1].repeat(len(neg_q), 1), neg_embeds, dim=-1
                        )
                        neg_sim = torch.clamp((neg_sim_raw + 1) / 2, 0.0, 1.0)

                        # 加权 MSE（local 更重）
                        weights = torch.tensor(
                            [local_weight if t == "local" else 1.0 for t in neg_types],
                            device=device,
                            dtype=neg_sim.dtype,
                        )
                        excess = torch.relu(torch.abs(neg_sim - 0.0) - 0.1)
                        errors = excess
                        weighted_mse = (weights * errors).sum() / (weights.sum() + 1e-8)

                        # hinge：local / unrelated 分别使用更严格的上界
                        margins = torch.tensor(
                            [local_margin if t == "local" else 0.4 for t in neg_types],
                            device=device,
                            dtype=neg_sim.dtype,
                        )
                        hinge = torch.relu(neg_sim - margins).mean()

                        neg_regression_loss += weighted_mse + hinge_weight * hinge
                        num_negs += 1
                if num_negs > 0:
                    neg_regression_loss /= num_negs
                else:
                    neg_regression_loss = torch.tensor(0.0, device=device)

                loss = contrastive_loss + regression_weight * regression_loss + regression_weight * neg_regression_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                correct_predictions += self.count_correct_predictions_contrastive(
                    anchor_embeds, positive_embeds, negative_questions_list, anchor_questions, device
                )
                total_predictions += len(anchor_embeds)

            avg_loss = total_loss / len(train_loader)
            accuracy = correct_predictions / total_predictions
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")

            if val_loader:
                val_loss, val_accuracy, pos_sims, neg_sims, gen_scores, below_05_ratio = self.evaluate(
                    val_loader, device, temperature, regression_weight=regression_weight
                )
                print(
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | "
                    f"Avg Gen Score: {np.mean(gen_scores):.4f} | Below 0.5 Ratio: {below_05_ratio:.3f}"
                )

                # 使用 below_05_ratio 作为早停指标，越大越好
                if below_05_ratio > best_ratio + min_delta:
                    best_ratio = below_05_ratio
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(self.classifier.state_dict())
                    similarity_distributions = {"positive": pos_sims, "negative": neg_sims}
                    print(f"Best model saved with below_05_ratio: {best_ratio:.3f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1} (best below_05_ratio: {best_ratio:.3f})")
                        if best_model_weights is not None:
                            self.classifier.load_state_dict(best_model_weights)
                        break
            print("---")

        if similarity_distributions:
            with open("models/similarity_distributions_v6A.json", "w") as f:
                json.dump(similarity_distributions, f, indent=2)

    def evaluate(self, data_loader, device, temperature=0.07, regression_weight=0.8):
        self.classifier.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        positive_similarities = []
        negative_similarities = []
        generalization_scores = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                anchor_questions = [item["anchor"]["question"] for item in batch]
                positive_questions = [item["positive"]["question"] for item in batch]
                negative_questions_list = [item["negative"] for item in batch]
                sim_labels = torch.tensor(
                    [item["similarity_label"] for item in batch], dtype=torch.float32
                ).to(device)

                anchor_embeds, positive_embeds = self.classifier(anchor_questions, positive_questions)

                contrastive_loss = self.info_nce_loss(
                    anchor_embeds, positive_embeds, negative_questions_list, anchor_questions, device, temperature
                )

                pos_sim_raw = torch.nn.functional.cosine_similarity(anchor_embeds, positive_embeds, dim=-1)
                pos_sim = torch.clamp((pos_sim_raw + 1) / 2, 0.0, 1.0)
                regression_loss = self.margin_mse(pos_sim, sim_labels, margin=0.05, power=1)

                # 负样本回归 + hinge
                neg_regression_loss = 0.0
                num_negs = 0
                local_weight = 3.0
                local_margin = 0.15
                hinge_weight = 1.0
                for i, negs in enumerate(negative_questions_list):
                    if negs:
                        neg_q = [n["question"] for n in negs]
                        neg_types = [n.get("neg_type", "") for n in negs]
                        _, neg_embeds = self.classifier([anchor_questions[i]] * len(neg_q), neg_q)
                        neg_sim_raw = torch.nn.functional.cosine_similarity(
                            anchor_embeds[i : i + 1].repeat(len(neg_q), 1), neg_embeds, dim=-1
                        )
                        neg_sim = torch.clamp((neg_sim_raw + 1) / 2, 0.0, 1.0)

                        weights = torch.tensor(
                            [local_weight if t == "local" else 1.0 for t in neg_types],
                            device=device,
                            dtype=neg_sim.dtype,
                        )
                        excess = torch.relu(torch.abs(neg_sim - 0.0) - 0.1)
                        errors = excess
                        weighted_mse = (weights * errors).sum() / (weights.sum() + 1e-8)

                        margins = torch.tensor(
                            [local_margin if t == "local" else 0.4 for t in neg_types],
                            device=device,
                            dtype=neg_sim.dtype,
                        )
                        hinge = torch.relu(neg_sim - margins).mean()

                        neg_regression_loss += weighted_mse + hinge_weight * hinge
                        num_negs += 1
                if num_negs > 0:
                    neg_regression_loss /= num_negs
                else:
                    neg_regression_loss = torch.tensor(0.0, device=device)

                loss = contrastive_loss + regression_weight * regression_loss + regression_weight * neg_regression_loss

                positive_similarities.extend(pos_sim.cpu().numpy().tolist())

                # 负样本相似度（取每个 anchor 的最大值）
                max_neg_sim = torch.full_like(pos_sim, float("-inf"))
                for i, negs in enumerate(negative_questions_list):
                    if negs:
                        neg_q = [n["question"] for n in negs]
                        _, neg_embeds = self.classifier([anchor_questions[i]], neg_q)
                        neg_sim_raw = torch.nn.functional.cosine_similarity(
                            anchor_embeds[i : i + 1], neg_embeds, dim=-1
                        )
                        neg_sim = torch.clamp((neg_sim_raw + 1) / 2, 0.0, 1.0)
                        max_neg_sim[i] = neg_sim.max()
                negative_similarities.extend(max_neg_sim.cpu().numpy().tolist())

                # generalization 分数（只统计 generalization_positive）
                for idx, item in enumerate(batch):
                    if item["sample_type"] == "generalization_positive":
                        generalization_scores.append(pos_sim[idx].item())

                total_loss += loss.item()
                correct_predictions += self.count_correct_predictions_contrastive(
                    anchor_embeds, positive_embeds, negative_questions_list, anchor_questions, device
                )
                total_predictions += len(anchor_embeds)

        # 统计无关样本相似度 < 0.5 的比例，帮助标定
        neg_sims_np = np.array(negative_similarities)
        below_05_ratio = float((neg_sims_np < 0.5).mean()) if neg_sims_np.size > 0 else 0.0
        print(f"[Eval] Unrelated pairs with sim < 0.5: {below_05_ratio:.3f}")

        return (
            total_loss / len(data_loader),
            correct_predictions / total_predictions,
            positive_similarities,
            negative_similarities,
            generalization_scores,
            below_05_ratio,  # 新增：返回负样本相似度 < 0.5 的比例
        )

    def info_nce_loss(
        self,
        anchor_embeds,
        positive_embeds,
        negative_questions_list,
        anchor_questions,
        device,
        temperature=0.07,
    ):
        batch_size = anchor_embeds.size(0)
        pos_sim = torch.nn.functional.cosine_similarity(anchor_embeds, positive_embeds, dim=-1) / temperature
        logits_list = [pos_sim.unsqueeze(-1)]

        # 显式负样本
        max_negatives = max(len(neg_list) for neg_list in negative_questions_list) if negative_questions_list else 0
        if max_negatives > 0:
            all_neg_questions = []
            for neg_idx in range(max_negatives):
                for i in range(batch_size):
                    if neg_idx < len(negative_questions_list[i]):
                        all_neg_questions.append(negative_questions_list[i][neg_idx]["question"])
                    else:
                        all_neg_questions.append("")
            if all_neg_questions:
                neg_inputs = self.classifier.tokenizer(
                    all_neg_questions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
                neg_outputs = self.classifier.model(**neg_inputs)
                
                # 使用 Mean Pooling（与 _encode 方法保持一致）
                neg_token_embeddings = neg_outputs.last_hidden_state
                neg_attention_mask = neg_inputs['attention_mask']
                neg_input_mask_expanded = neg_attention_mask.unsqueeze(-1).expand(neg_token_embeddings.size()).float()
                neg_sum_embeddings = torch.sum(neg_token_embeddings * neg_input_mask_expanded, 1)
                neg_sum_mask = torch.clamp(neg_input_mask_expanded.sum(1), min=1e-9)
                neg_embeds_all = neg_sum_embeddings / neg_sum_mask
                
                neg_embeds_batch = neg_embeds_all.view(batch_size, max_negatives, -1)
                for neg_idx in range(max_negatives):
                    neg_embeds = neg_embeds_batch[:, neg_idx, :]
                    neg_sim = torch.nn.functional.cosine_similarity(anchor_embeds, neg_embeds, dim=-1) / temperature
                    logits_list.append(neg_sim.unsqueeze(-1))

        # batch 内 hard negatives
        if batch_size > 1:
            expanded_anchor = anchor_embeds.unsqueeze(1).expand(-1, batch_size, -1)
            expanded_pos = positive_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            batch_sim = torch.nn.functional.cosine_similarity(expanded_anchor, expanded_pos, dim=-1) / temperature
            hard_negs = batch_sim[mask].view(batch_size, batch_size - 1)
            logits_list.append(hard_negs)

        logits = torch.cat(logits_list, dim=-1)
        target = torch.zeros(batch_size, dtype=torch.long, device=device)
        return nn.CrossEntropyLoss()(logits, target)

    def count_correct_predictions_contrastive(
        self, anchor_embeds, positive_embeds, negative_questions_list, anchor_questions, device
    ):
        # 与训练时保持一致：归一化相似度
        pos_sim_raw = torch.nn.functional.cosine_similarity(anchor_embeds, positive_embeds, dim=-1)
        pos_sim = torch.clamp((pos_sim_raw + 1) / 2, 0.0, 1.0)
        max_neg_sim = torch.full_like(pos_sim, float("-inf"))
        for i, negs in enumerate(negative_questions_list):
            if negs:
                neg_q = [n["question"] for n in negs]
                _, neg_embeds = self.classifier([anchor_questions[i]], neg_q)
                neg_sim_raw = torch.nn.functional.cosine_similarity(
                    anchor_embeds[i : i + 1], neg_embeds, dim=-1
                )
                neg_sim = torch.clamp((neg_sim_raw + 1) / 2, 0.0, 1.0)
                max_neg_sim[i] = neg_sim.max()
        return (pos_sim > max_neg_sim).sum().item()

    def save_model(self, model: PureXLMSimilarityClassifier, save_path: str):
        """
        保存格式与 SERAC 推理端严格对齐：
        - 使用 'model_state_dict' 作为键，方便 serac_main.py 直接加载到底层 classifier 上。
        """
        state_dict = {
            "model_state_dict": model.model.state_dict(),
            "model_name": "xlm-roberta-base",
            "model_config": model.model.config.to_dict(),
            "similarity_logic": {
                "method": "cosine_similarity",
                "temperature": 0.05,
            },
        }
        torch.save(state_dict, save_path)
        model.model.save_pretrained(save_path.replace(".pth", ""))
        model.tokenizer.save_pretrained(save_path.replace(".pth", ""))
        print(f"Model saved to {save_path}")


# ========================================
# 主函数
# ========================================
def main():
    print("加载数据...")
    with open("data/Bi-ZsRE-data/zsre_mend_train_10000.json", "r", encoding="utf-8") as f:
        en_data = json.load(f)
    with open("data/Bi-ZsRE-data/zsre_mend_train_10000_chinese.json", "r", encoding="utf-8") as f:
        zh_data = json.load(f)

    print("生成训练数据...")
    generator = MultilingualDataGenerator()
    training_data = generator.generate_training_data(en_data, zh_data, max_samples_per_type=20000)

    train_size = int(0.9 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]

    train_dataset = SimilarityDataset(train_data)
    val_dataset = SimilarityDataset(val_data)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    print("创建纯 XLM 相似度分类器...")
    classifier = PureXLMSimilarityClassifier()
    for param in classifier.model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(classifier.parameters(), lr=2e-5, weight_decay=0.01)

    print("开始训练...")
    trainer = MemorylessClassifierTrainer(classifier, optimizer)
    trainer.train_without_memory(
        train_loader,
        val_loader,
        epochs=100,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        temperature=0.05,
        regression_weight=0.8,
    )

    os.makedirs("models", exist_ok=True)
    trainer.save_model(
        classifier,
        "models/6-11-C-1multilingual_similarity_classifier_purexlm_v6A.pth",
    )
    print("训练完成！")


if __name__ == "__main__":
    main()



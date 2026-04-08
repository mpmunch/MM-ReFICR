from gritlm import GritLM
from scipy.spatial.distance import cosine
import json
import re
from pathlib import Path
from jsonargparse import CLI
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np
from transformers import set_seed, AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType,PeftModel
import os
from typing import Dict, List, Optional, Tuple
from utils import search_number,extract_movie_name, recall_score, add_roles, is_float

from rank_bm25 import BM25Okapi
from llm2vec import LLM2Vec

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _normalize_item_title(text: str) -> str:
    text = text.lower()
    text = text.replace("&", " and ")
    text = text.replace("-", " ")
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_variants(title: str) -> List[str]:
    base = _normalize_item_title(title)
    no_year = re.sub(r"\s*\(?\d{4}\)?\s*$", "", title).strip()
    no_year_norm = _normalize_item_title(no_year)
    variants = [x for x in [base, no_year_norm] if x]
    return list(dict.fromkeys(variants))


def _load_image_payload(image_embeddings_path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    payload = torch.load(image_embeddings_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected image embedding payload dict, got {type(payload)}")

    required = ["db_titles", "db_embeddings", "db_found_mask"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing keys in image embedding payload: {missing}")

    db_embeddings = payload["db_embeddings"].to(torch.float32).cpu()
    db_found_mask = payload["db_found_mask"].to(torch.bool).cpu()

    title_to_db_idx: Dict[str, int] = {}
    for idx, title in enumerate(payload["db_titles"]):
        if not isinstance(title, str):
            continue
        for key in _title_variants(title):
            if key not in title_to_db_idx:
                title_to_db_idx[key] = idx

    found = int(db_found_mask.sum().item())
    total = int(db_found_mask.numel())
    coverage = 100.0 * found / max(total, 1)
    print(
        f"Loaded image payload: normalized_titles={len(title_to_db_idx)} | db_found={found}/{total} ({coverage:.2f}%)"
    )
    return db_embeddings, db_found_mask, title_to_db_idx


def _build_item_image_tensors(
    item_titles: List[str],
    db_embeddings: torch.Tensor,
    db_found_mask: torch.Tensor,
    title_to_db_idx: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    image_emb = torch.zeros((len(item_titles), db_embeddings.size(1)), dtype=torch.float32)
    image_mask = torch.zeros(len(item_titles), dtype=torch.bool)

    missing_title = 0
    missing_image = 0
    for i, title in enumerate(item_titles):
        db_idx = None
        for key in _title_variants(title):
            if key in title_to_db_idx:
                db_idx = title_to_db_idx[key]
                break
        if db_idx is None:
            missing_title += 1
            continue
        if not db_found_mask[db_idx]:
            missing_image += 1
            continue
        image_emb[i] = db_embeddings[db_idx]
        image_mask[i] = True

    found = int(image_mask.sum().item())
    total = int(image_mask.numel())
    coverage = 100.0 * found / max(total, 1)
    print(
        f"Item image match coverage: {found}/{total} ({coverage:.2f}%) | no_title_match={missing_title} | no_image={missing_image}"
    )
    return image_emb, image_mask


def _find_latest_checkpoint_dir(target_model_path: str) -> Optional[str]:
    root = Path(target_model_path)
    if not root.exists() or not root.is_dir():
        return None

    ckpts = []
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        suffix = p.name.split("checkpoint-")[-1]
        if suffix.isdigit():
            ckpts.append((int(suffix), p))
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


def _build_linear_from_state(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.nn.Linear:
    out_features, in_features = weight.shape
    layer = torch.nn.Linear(in_features, out_features, bias=bias is not None)
    layer.weight.data.copy_(weight)
    if bias is not None:
        layer.bias.data.copy_(bias)
    layer.eval()
    return layer


def _load_image_projection_from_non_lora(target_model_path: str) -> Optional[torch.nn.Linear]:
    non_lora_path = os.path.join(target_model_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        return None

    state = torch.load(non_lora_path, map_location="cpu")
    if not isinstance(state, dict):
        return None

    w = state.get("image_projection.weight", None)
    b = state.get("image_projection.bias", None)
    if w is None:
        return None

    print("Loaded image projection from non_lora_trainables.bin")
    return _build_linear_from_state(w.to(torch.float32), b.to(torch.float32) if b is not None else None)


def _load_image_projection_from_checkpoint(checkpoint_dir: str) -> Optional[torch.nn.Linear]:
    if checkpoint_dir is None:
        return None

    index_path = os.path.join(checkpoint_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(index_path):
        return None

    with open(index_path, "r", encoding="utf-8") as f:
        idx = json.load(f)

    weight_map = idx.get("weight_map", {})
    w_key = "image_projection.weight"
    b_key = "image_projection.bias"
    if w_key not in weight_map:
        return None

    shard_name = weight_map[w_key]
    shard_path = os.path.join(checkpoint_dir, shard_name)
    if not os.path.exists(shard_path):
        return None

    print(f"Loading projection shard: {shard_path}")
    shard_state = torch.load(shard_path, map_location="cpu")
    w = shard_state.get(w_key, None)
    b = shard_state.get(b_key, None)
    if w is None:
        return None

    print(f"Loaded image projection from checkpoint: {checkpoint_dir}")
    return _build_linear_from_state(w.to(torch.float32), b.to(torch.float32) if b is not None else None)
        
#merge the model weights
def apply_lora(base_model_path, target_model_path):

    # base model
    #tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, padding_side="right")
    model = GritLM(base_model_path, low_cpu_mem_usage=True, torch_dtype="auto")
    
    if os.path.exists(os.path.join(target_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(target_model_path, 'non_lora_trainables.bin'), map_location='cpu')
        print(non_lora_trainables)
        model.load_state_dict(non_lora_trainables, strict=False)

    #peft model
    print(f"Loading LoRA weights from {target_model_path}")
    lora_model = PeftModel.from_pretrained(model.model, target_model_path)
    print(f"Merging weights")
    model.model = lora_model.merge_and_unload()
    return model

def gritlm_instruction(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"


def get_instruction(data, task_type, gen_instr):

    output = []
    for example in data:
        context = example["context"]
        
        if task_type == "Ranking":
        
            num = 10
            cand_dict = example["cand_list"]
            top_k_items = {k: cand_dict[k] for k in list(cand_dict)[:10]}
            cand_items = ""
            for key, value in top_k_items.items():
                cand_items += f"[{str(key)}] {str(value)}\n"        
            
            rag_kg_conv = ' '.join(example["re_kg"]["context"][-4:])
            rag_kg_target = example["re_kg"]["target"]
            if len(rag_kg_target) > 0:
                target = ', '.join(rag_kg_target)
            else:
                target = rag_kg_target[0]
            retrieved_kg = f"Users with intentions similar to the current user were recommended {target} by the system. The refered content is: {rag_kg_conv[-512:]}"


            pre_prompt = gen_instr.format(cand_items,context[-512:],retrieved_kg,num)
            #pre_prompt = gen_instr.format(cand_items,context[-512:],num)
            #pre_prompt = gen_instr.format(context[-512:],retrieved_kg,cand_items)

        if task_type == "Dialoge_Manage":
            pre_prompt = gen_instr.format(context[-516:])

        if task_type == "Response_Gen":
            recommend_item = " ".join(example["rec"])
            pre_prompt = gen_instr.format(context[-516:],recommend_item)

        #print("pre_prompt:",pre_prompt)
        messages = [{ 
                        "role":"user",
                        "content":pre_prompt}]

        output.append(messages)

    return output

def BM25_retrieval(corpus, queries):
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    ranked_list = []
    for query in tqdm(queries):
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        top50_indices = doc_scores.argsort()[::-1][:50]

        print("top50_indices:",top50_indices)
        ranked_list.append(top50_indices)

    return ranked_list


def LLM2Vec_retrieval(queries,documents,rec_lists):

    tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",)

    model = PeftModel.from_pretrained(
        model,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",)

    model = model.merge_and_unload()  # This can take several minutes on cpu

    # Loading supervised model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
    model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer,pooling_mode="mean", max_length=512)

    # Encoding documents. Instruction are not required for documents
    d_reps = l2v.encode(documents)
    print("d_reps shape:", d_reps.shape)

    # Encoding queries using instructions
    instruction = ("Retrieve relevant items based on user conversation history:")

    new_queries = [[instruction,query] for query in queries]

    num_slice = 4
    step = int(len(new_queries) / num_slice) + 1
    print('query_step:',step)
    rank = []

    for i in range(0,len(new_queries),step):
        queries_slice = new_queries[i : i + step]
        rec_lists_slice = rec_lists[i : i + step]

        assert len(queries_slice) == len(rec_lists_slice)
            
        q_reps = l2v.encode(queries_slice)
        print("q_reps shape:", q_reps.shape)

        # Compute cosine similarity
        q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1)
        d_reps_norm = torch.nn.functional.normalize(d_reps, p=2, dim=1)
        cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

        topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=50,dim=-1)
        rank_slice = topk_sim_indices.tolist()
        rank += rank_slice
    print('length rank:',len(rank))

    return rank



def main(mode:str=None, tag:str=None, query_instr:str=None, doc_instr:str=None, gen_instr:str=None,from_json:str=None, db_json:str=None, embeddings_path:str=None, base_model_path:str="GritLM/GritLM-7B",
    target_model_path:str=None, to_json:str=None, stored_cand_lst:bool=True, is_lora:bool=True, batch_size:int=8,
    use_image_features: bool=False, image_embeddings_path: str=None, image_fusion_weight: float=0.2,
    image_projection_checkpoint: str=None):

    
    set_seed(123)
    
    if is_lora:
        model = apply_lora(base_model_path,target_model_path)
    else:
        model = GritLM("GritLM/GritLM-7B", torch_dtype="auto")

    with open(from_json) as fd:
        lines = fd.readlines()
        data = [json.loads(line) for line in lines]
        print(len(data))

    if mode == 'embedding':

        with open(db_json) as fi:
            db = json.load(fi)
        print(len(db))

        queries = [example['context'][-512:] for example in data]
        print('queries length:',len(queries))


        if tag == "Conv2Item":

            all_names = list(db.keys())
            name2id = {all_names[index]: index for index in range(len(all_names))}
            print("name2id:",len(name2id))
            id2name = {v:k for k,v in name2id.items()}


            docs = list(db.values())
            docs = [doc[:1024] for doc in docs]
            docs_len = [len(doc) for doc in docs]
            print("max docs:",np.max(docs_len))
            print("mean docs:",np.mean(docs_len))
            print("min docs:",np.min(docs_len))
            print('doc length:',len(docs))

            image_projection_layer = None
            db_image_embeddings = None
            db_image_found_mask = None
            title_to_db_idx = None
            if use_image_features:
                if image_embeddings_path is None:
                    raise ValueError("use_image_features=True requires image_embeddings_path")

                db_image_embeddings, db_image_found_mask, title_to_db_idx = _load_image_payload(image_embeddings_path)

                image_projection_layer = _load_image_projection_from_non_lora(target_model_path)
                if image_projection_layer is None:
                    if image_projection_checkpoint is None:
                        image_projection_checkpoint = _find_latest_checkpoint_dir(target_model_path)
                    image_projection_layer = _load_image_projection_from_checkpoint(image_projection_checkpoint)

                if image_projection_layer is None:
                    raise ValueError(
                        "Could not load image projection weights. Provide --image_projection_checkpoint "
                        "pointing to a trainer checkpoint directory containing pytorch_model.bin.index.json"
                    )

            if os.path.exists(embeddings_path):
                print("loading embeddings form file")
                d_rep = torch.load(embeddings_path)
            else:
                print(f"Encoding {len(docs)} docs in batches of {batch_size} ...")
                all_chunks = []
                for i in tqdm(range(0, len(docs), batch_size)):
                    batch_docs = docs[i:i+batch_size]
                    with torch.inference_mode():
                        batch_rep = model.encode(batch_docs, instruction=gritlm_instruction(doc_instr))

                    # convert to CPU tensor
                    if isinstance(batch_rep, torch.Tensor):
                        batch_rep_cpu = batch_rep.detach().cpu()
                    else:
                        batch_rep_cpu = torch.from_numpy(batch_rep).cpu()

                    all_chunks.append(batch_rep_cpu)

                    torch.cuda.empty_cache()

                d_rep = torch.cat(all_chunks, dim=0)

                print('document shape:', d_rep.shape)
                torch.save(d_rep, embeddings_path)
                print("saving embeddigns to file ...") 

            if isinstance(d_rep, torch.Tensor):
                d_rep_t = d_rep.detach().cpu().to(torch.float32)
            else:
                d_rep_t = torch.from_numpy(d_rep).cpu().to(torch.float32)

            if use_image_features:
                item_image_emb, item_image_mask = _build_item_image_tensors(
                    all_names,
                    db_image_embeddings,
                    db_image_found_mask,
                    title_to_db_idx,
                )
                with torch.inference_mode():
                    image_reps = image_projection_layer(item_image_emb.to(dtype=d_rep_t.dtype))
                    image_reps = F.normalize(image_reps, p=2, dim=1)
                    d_rep_t = F.normalize(d_rep_t, p=2, dim=1)
                    mask = item_image_mask.to(dtype=d_rep_t.dtype).unsqueeze(-1)
                    d_rep_t = d_rep_t + mask * image_fusion_weight * (image_reps - d_rep_t)
                    d_rep_t = F.normalize(d_rep_t, p=2, dim=1)
                print(f"Applied multimodal fusion with alpha={image_fusion_weight}")

            rec_lists = []
            for example in tqdm(data):
                lst = []
                for item in example['rec']:
                    if item in all_names:
                        lst.append(name2id[item])
                    else:
                        for name, desc in db.items():
                            if extract_movie_name(name) == extract_movie_name(item):
                                lst.append(name2id[name])
                lst = list(set(lst))
                rec_lists.append(lst)

            num_slice = 4
            step = int(len(queries) / num_slice) + 1
            print('query_step:',step)
            rank = []
            score_lst = []

            for i in range(0,len(queries),step):
                queries_slice = queries[i : i + step]
                rec_lists_slice = rec_lists[i : i + step]

                assert len(queries_slice) == len(rec_lists_slice)
                
                q_rep = model.encode(queries_slice, instruction=gritlm_instruction(query_instr), batch_size=batch_size)

                # convert embeddings to CPU tensors
                if isinstance(q_rep, torch.Tensor):
                    q_rep_t = q_rep.detach().cpu()
                else:
                    q_rep_t = torch.from_numpy(q_rep).cpu()

                print('queries shape:', q_rep_t.shape) 

                cos_sim = F.cosine_similarity(
                    q_rep_t.unsqueeze(1),
                    d_rep_t.unsqueeze(0),
                    dim=-1
                )
                cos_sim = torch.where(
                    torch.isnan(cos_sim),
                    torch.full_like(cos_sim,0),
                    cos_sim
                )
                print("cos_sim shape:",cos_sim.shape)
                print("cos_sim:",cos_sim)

                topk_sim_values,topk_sim_indices = torch.topk(cos_sim,k=50,dim=-1)
                rank_slice = topk_sim_indices.tolist()
                rank += rank_slice
                print('length rank:',len(rank))


            print('length rank:',len(rank))
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))
            
            if stored_cand_lst:

                for i in range(len(rank)):

                    ranked_list = {j:id2name[j] for j in rank[i]}

                    data[i]["rec_id"] = rec_lists[i]
                    data[i]["cand_list"] = ranked_list

                with open(to_json,"w",encoding="utf-8") as fwr:
                    for example in data:
                        fwr.write(json.dumps(example))
                        fwr.write("\n")

        if tag == "Conv2Conv":

            #conv_docs = [example_k["context"] for example_k in db.values()]
            #conv_docs = [conv_doc[:1024] for conv_doc in conv_docs]
            
            conv_docs = []
            for dict_conv in db.values():
                context = add_roles(dict_conv['context'])
                conv_docs.append(context)
            print("conv_doc:", conv_docs[0])    
            print('length of conv_docs:',len(conv_docs))

            if os.path.exists(embeddings_path):
                print("loading embeddings form file")
                conv_d_rep = torch.load(embeddings_path)
            else:
                conv_d_rep = model.encode(conv_docs, instruction=gritlm_instruction(doc_instr), batch_size=batch_size)
                print('conv doc shape:',torch.from_numpy(conv_d_rep).shape)
                torch.save(conv_d_rep, embeddings_path)
                print("saving embeddigns to file ...")

            conv_q_rep = model.encode(queries, instruction=gritlm_instruction(query_instr), batch_size=batch_size)
            print('conv queries shape:',torch.from_numpy(conv_q_rep).shape)
            #normalize
            conv_d_rep = F.normalize(torch.from_numpy(conv_d_rep), p=2, dim=1)
            conv_q_rep = F.normalize(torch.from_numpy(conv_q_rep), p=2, dim=1)

            #compute similarity
            conv_cos_similarities = torch.mm(conv_q_rep, conv_d_rep.t())
            #print("conv_d_rep:",conv_d_rep.shape)
            #print("conv_q_rep:",conv_q_rep.shape)
            #print("conv_cos_similarities:",conv_cos_similarities.shape)

            #cos_similarities = conv_cos_similarities
            #topk_conv_values,topk_conv_indices = torch.topk(cos_similarities,k=5,dim=-1)
            #conv_indices = topk_conv_indices.tolist()

            #print(recall_score(sim_conv,conv_indices,ks=[1,5,10,20,50]))
            #print("cos_similarities:",cos_similarities.shape)
            #print("topk_conv_values:",topk_conv_values.shape)
            #print("topk_conv_indices:",topk_conv_indices.shape)
            #for i in range(len(conv_indices)):
            #    re_kg = [db[str(conv_indices[i][j])]['target'] for j in range(5)]
            #    data[i]["re_kg"] = re_kg

            #with open(to_json,"w",encoding="utf-8") as fr:
            #    for example in data:
            #        fr.write(json.dumps(example))
            #        fr.write("\n")


            cos_similarities = conv_cos_similarities
            topk_conv_values,topk_conv_indices = torch.topk(cos_similarities,k=1,dim=-1)
            conv_indices = topk_conv_indices.tolist()
            print("cos_similarities:",cos_similarities.shape)
            print("topk_conv_values:",topk_conv_values.shape)
            print("topk_conv_indices:",topk_conv_indices.shape)
            for i in range(len(conv_indices)):
                #print(conv_indices[i][0])
                re_kg = db[str(conv_indices[i][0])]
                sim_value = topk_conv_values[i][0]
                #print("re_kg:",re_kg)
                #print("sim_value:",sim_value)
                data[i]["re_kg"] = re_kg
                data[i]["sim_value"] = sim_value.item()

            with open(to_json,"w",encoding="utf-8") as fr:
                for example in data:
                    fr.write(json.dumps(example))
                    fr.write("\n")

    if mode == "generation":
        outputs = get_instruction(data,tag,gen_instr)

        rank = []
        rank_len = []
        pred = []
        for messages in tqdm(outputs):

            encoded = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            encoded = encoded.to(model.device)
            gen = model.generate(encoded, max_new_tokens=1024, do_sample=False, pad_token_id=2)
            decoded = model.tokenizer.batch_decode(gen)
            print(decoded[0].encode("utf-8").decode("latin1"))
            generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","")
            pred.append(generated)

            if tag == "Ranking":

                generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").split("\n")
                #print("generated:",generated)
                clean_generated_rank = []

                """for each_rank in generated:
                    if len(search_number(each_rank))==0:
                        continue
                    clean_generated_rank.append(int(search_number(each_rank)))"""

                rank_score = {}
                for each_rank in generated:
                    result = each_rank.split(",")
                    #print("result:",result)
                    if len(search_number(result[0]))==0:
                        continue

                    """if is_float(result[-1]):
                        identity = int(search_number(result[0]))
                        rank_score[identity] = float(result[-1].strip())
                        #print("rank_score:",rank_score)
                        clean_generated_rank = sorted(rank_score, key=rank_score.get, reverse=True)

                    else:
                        clean_generated_rank.append(int(search_number(each_rank)))"""

                    clean_generated_rank.append(int(search_number(each_rank)))

                print("clean_generated_rank:",clean_generated_rank)
                rank.append(clean_generated_rank)
                rank_len.append(len(clean_generated_rank))

            # generated = decoded[0].split("<|assistant|>\n")[-1].replace("</s>","").replace("\n","").strip()
            # pred.append(generated)


        if tag == "Ranking":
            print("max rank:",np.max(rank_len))
            print("mean rank:",np.mean(rank_len))
            print("min rank:",np.min(rank_len))
            print('length rank:',len(rank))

            rec_lists = [example["rec_id"] for example in data]
            assert len(rec_lists) == len(rank)
            print(recall_score(rec_lists,rank,ks=[1,5,10,20,50]))


        if tag == "Dialoge_Manage":
            
            assert len(pred) == len(data)

            with open(to_json,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                    #print("pred[e_id]:", pred[e_id])
                    data[e_id]["action"] = pred[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

        if tag == "Response_Gen":

            assert len(pred) == len(data)

            with open(to_json,"w",encoding="utf-8") as fout:
                for e_id in range(len(data)):
                   
                    if len(data[e_id]["rec"]) == 0:
                        data[e_id]["rec_tag"] = 0
                    else:
                        data[e_id]["rec_tag"] = 1

                    data[e_id]["pred"] = pred[e_id]
                    fout.write(json.dumps(data[e_id],ensure_ascii=False))
                    fout.write("\n")

if __name__ == '__main__':
    CLI(main)

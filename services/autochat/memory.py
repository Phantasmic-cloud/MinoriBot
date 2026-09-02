try:
    from .utils import *
except ImportError:
    from utils import *
import chromadb
import uuid
import time
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

TEXT_EMB_DIM = config.get('text_embed_dim')

@dataclass
class EventMemory:
    id: str
    text: str
    type: str
    weight: int
    created_at: float
    distance: float = 0.0
    adjusted_distance: float = 0.0
    time_penalty: float = 0.0

@dataclass
class UserMemory:
    names: List[str] = field(default_factory=list)
    profile: str = ""
    recent_events: List[Tuple[float, str]] = field(default_factory=list)

@dataclass
class SelfMemory:
    id: str
    text: str
    time: float


class MemorySystem:
    def __init__(self, data_dir: str, group_id: int):
        os.makedirs(data_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=os.path.join(data_dir, f"memory_chromadb_{group_id}"))
        self.em_collection = self.client.get_or_create_collection(name="event_memroy")
        self.file_db = get_file_db(os.path.join(data_dir, f"memory_{group_id}.json")) 

    def em_add(self, text: str, embedding: List[float], initial_weight: int) -> str:
        """
        添加一条新的事件记忆，默认为短期记忆。  

        参数:
            text (str): 文本。
            embedding (List[float]): 文本对应的embedding向量。
            initial_weight (int): 记忆的初始权重。
        返回:
            str: 该条记忆的唯一ID。
        """
        assert len(embedding) == TEXT_EMB_DIM, f"Embedding维度应为 {TEXT_EMB_DIM}，但收到 {len(embedding)}"
        memory_id = str(uuid.uuid4())
        self.em_collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[{
                "text": text,
                "type": "short_term",
                "weight": initial_weight,
                "created_at": time.time(),
            }]
        )
        info(f"成功添加事件记忆 {memory_id} 内容: \"{text}\"")
        return memory_id

    def em_increase_weight(self, memory_id: str, weight_increase: int, threshold: int):
        """
        为指定ID的事件记忆增加权重。当权重达到阈值时，转换为长期记忆。

        参数:
            memory_id (str): 要增加权重的记忆ID。
            weight_increase (int): 每次增加的权重量。
            threshold (int): 转换为长期记忆的权重阈值。
        """
        # 获取当前记忆的元数据
        memory = self.em_collection.get(ids=[memory_id], include=["metadatas"])
        if not memory['ids']:
            raise Exception(f"未找到ID为 {memory_id} 的事件记忆")

        current_meta = memory['metadatas'][0]
        current_weight = current_meta.get('weight', 0)
        
        # 更新权重
        new_weight = current_weight + weight_increase
        current_meta['weight'] = new_weight
        
        info(f"事件记忆 {memory_id} 权重 {current_weight} -> {new_weight}")

        # 检查是否达到阈值
        if new_weight >= threshold and current_meta.get('type') == 'short_term':
            current_meta['type'] = 'long_term'
            info(f"记忆 {memory_id} 已转换为长期记忆。")

        # 将更新后的元数据写回数据库
        self.em_collection.update(ids=[memory_id], metadatas=[current_meta])

    def em_query(
        self, 
        query_embeddings: List[List[float]], # 修改：支持传入多个embedding
        n_results: int, 
        memory_type: str,
        time_decay_rate: float = 0.0,
    ) -> List[EventMemory]:
        """
        查询最相关的记忆，支持混合检索（同时传入摘要向量和最新消息向量）。
        
        参数:
            query_embedding (List[float]): 用于查询的embedding向量。
            n_results (int): 希望返回的结果数量。
            memory_type (str): 查询类型, 可选 "all", "short_term", "long_term"。
            time_decay_rate (float): 短期记忆时间衰减率（每小时增加的距离惩罚）。

        返回:
            List[EMQueryResult]: 相关记忆的查询结果列表
        """
        if n_results <= 0 or not query_embeddings:
            return []
        
        for emb in query_embeddings:
            assert len(emb) == TEXT_EMB_DIM, f"Embedding维度错误"

        where_clause = {}
        if memory_type in ["short_term", "long_term"]:
            where_clause = {"type": memory_type}

        # 稍微放大查询数量，以便混合去重
        query_n = n_results * 2

        # ChromaDB 支持一次查询多个向量
        results = self.em_collection.query(
            query_embeddings=query_embeddings,
            n_results=query_n,
            where=where_clause,
            include=["metadatas", "distances"]
        )

        # 结果去重与合并
        # results['ids'] 是一个列表的列表 [[id1, id2], [id3, id4]]
        unique_memories: Dict[str, EventMemory] = {}

        for i in range(len(query_embeddings)): # 遍历每一次查询（摘要查询、当前消息查询）
            ids = results['ids'][i]
            distances = results['distances'][i]
            metadatas = results['metadatas'][i]

            for j, mem_id in enumerate(ids):
                # 构造对象
                created_at = metadatas[j].get('created_at', 0)
                mem_type = metadatas[j].get('type', 'unknown')
                raw_distance = distances[j]
                
                # 计算时间惩罚
                adjusted_distance = raw_distance
                time_penalty = 0.0
                if mem_type == 'short_term':
                    time_elapsed_seconds = time.time() - created_at
                    time_elapsed_hours = time_elapsed_seconds / 3600
                    time_penalty = time_elapsed_hours * time_decay_rate
                    adjusted_distance += time_penalty

                # 如果ID已存在，取距离更小（更相关）的那次
                if mem_id in unique_memories:
                    if adjusted_distance < unique_memories[mem_id].adjusted_distance:
                        unique_memories[mem_id].distance = raw_distance
                        unique_memories[mem_id].adjusted_distance = adjusted_distance
                        unique_memories[mem_id].time_penalty = time_penalty
                else:
                    unique_memories[mem_id] = EventMemory(
                        id=mem_id,
                        text=metadatas[j].get('text', ''),
                        distance=raw_distance,
                        adjusted_distance=adjusted_distance,
                        type=mem_type,
                        weight=metadatas[j].get('weight', 0),
                        created_at=created_at,
                        time_penalty=time_penalty,
                    )

        # 转换为列表并排序
        final_results = list(unique_memories.values())
        final_results.sort(key=lambda x: x.adjusted_distance)
        
        return final_results[:n_results]

    def em_forget(self, forget_time: float, forget_prob: float):
        """
        遗忘指定时间之前的短期记忆，按概率遗忘。

        参数:
            forget_time (float): 遗忘时间点（时间戳），早于此时间的短期记忆将被考虑遗忘。
            forget_prob (float): 遗忘概率（0到1之间）。
        """
        candidate_memories = self.em_collection.get(
            where={
                "$and": [
                    {"type": "short_term"},
                    {"created_at": {"$lt": forget_time}}
                ]
            }
        )
        if not candidate_memories['ids']:
            return
        # 遍历候选记忆，根据概率决定哪些需要被遗忘
        ids_to_forget = []
        for memory_id in candidate_memories['ids']:
            if random.random() < forget_prob:
                ids_to_forget.append(memory_id)
        # 如果有需要遗忘的记忆，则执行删除操作
        if ids_to_forget:
            info(f"将遗忘 {len(ids_to_forget)} 条短期记忆")
            self.em_collection.delete(ids=ids_to_forget)
            info(f"成功遗忘短期记忆: {ids_to_forget}")
        else:
            info("没有短期记忆被遗忘")

    def um_update(
        self, 
        user_id: int, 
        new_names: List[str] = None,
        wrong_names: List[str] = None,
        profile_update: str = None, 
        event_update: str = None, 
        max_events: int = 5, 
        max_names: int = 5,
    ):
        """
        更新用户记忆（增量更新）。
        """
        ums = self.file_db.get('ums', {})
        uid_str = str(user_id)
        
        # 加载现有记忆或创建新的
        if uid_str in ums:
            current_um = UserMemory(**ums[uid_str])
            if not isinstance(current_um.names, list): current_um.names = list(current_um.names)
            if not isinstance(current_um.recent_events, list): current_um.recent_events = list(current_um.recent_events)
        else:
            current_um = UserMemory()

        updated = False
        
        # 1. 更新名字
        if wrong_names:
            for wrong_name in wrong_names:
                if wrong_name in current_um.names:
                    current_um.names.remove(wrong_name)
                    updated = True
                    info(f"移除用户 {user_id} 错误名字: {wrong_name}")
        if new_names:
            for new_name in new_names:
                if new_name in current_um.names:
                    continue
                current_um.names.append(new_name)
                current_um.names = current_um.names[-max_names:]
                updated = True
                info(f"更新用户 {user_id} 曾用名: {new_name}")
        
        # 2. 更新用户画像
        if profile_update and profile_update != current_um.profile:
            current_um.profile = profile_update
            updated = True
            info(f"更新用户 {user_id} 画像")

        # 3. 添加新事件 
        if event_update:
            current_um.recent_events.append((time.time(), event_update))
            current_um.recent_events = current_um.recent_events[-max_events:]
            updated = True
            info(f"更新用户 {user_id} 事件: {event_update}")

        if updated:
            ums[uid_str] = asdict(current_um)
            self.file_db.set('ums', ums)

    def um_get(self, user_id: int) -> UserMemory | None:
        ums = self.file_db.get('ums', {})
        if str(user_id) in ums:
            data = ums[str(user_id)]
            # 兼容旧数据结构（如果之前只存了text）
            if 'text' in data and 'profile' not in data:
                 return UserMemory(profile=data['text'])
            return UserMemory(**data)
        return None
    
    def um_query_uid_by_name_in_message(self, message: str) -> set[int]:
        """
        通过消息内容中出现的名字查询用户记忆对应的用户ID。
        """
        ums = self.file_db.get('ums', {})
        results = set()
        for uid_str, data in ums.items():
            user_memory = UserMemory(**data)
            for name in user_memory.names:
                if name in message:
                    results.add(int(uid_str))
                    break
        return results

    def sm_add(self, msg_id: int, text: str, keep_count: int):
        """
        更新自身对话记忆。

        参数:
            msgs (list[dict]): 新加入的对话消息列表。
            keep_count (int): 保留的消息数量。
        """
        sms = self.file_db.get('sms', [])
        sms.append({
            'id': str(msg_id),
            'text': text,
            'time': datetime.now().timestamp(),
        })
        sms = sms[-keep_count:]
        self.file_db.set('sms', sms)
        info(f"更新自身对话记忆，保留最近 {keep_count} 条消息")

    def sm_get(self) -> list[SelfMemory]:
        """
        获取自身对话记忆。

        返回:
            list[dict]: 对话消息列表。
        """
        return [SelfMemory(**sm) for sm in self.file_db.get('sms', [])]


